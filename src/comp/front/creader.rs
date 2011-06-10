// -*- rust -*-

import driver::session;
import front::ast;
import lib::llvm::False;
import lib::llvm::llvm;
import lib::llvm::mk_object_file;
import lib::llvm::mk_section_iter;
import middle::resolve;
import middle::walk;
import middle::metadata;
import middle::trans;
import middle::ty;
import back::x86;
import util::common;
import util::common::span;
import util::common::respan;
import util::common::a_bang;
import util::common::a_ty;
import util::common::may_begin_ident;

import std::str;
import std::uint;
import std::vec;
import std::ebml;
import std::fs;
import std::io;
import std::option;
import std::option::none;
import std::option::some;
import std::os;
import std::map::hashmap;


// Type decoding

// Compact string representation for ty::t values. API ty_str & parse_from_str
// (The second has to be authed pure.) Extra parameters are for converting
// to/from def_ids in the data buffer. Whatever format you choose should not
// contain pipe characters.

// Callback to translate defs to strs or back:
type str_def = fn(str) -> ast::def_id;

type pstate = rec(vec[u8] data, int crate,
                  mutable uint pos, uint len, ty::ctxt tcx);

type ty_or_bang = util::common::ty_or_bang[ty::t];

fn peek(@pstate st) -> u8 {
    ret st.data.(st.pos);
}
fn next(@pstate st) -> u8 {
    auto ch = st.data.(st.pos);
    st.pos = st.pos + 1u;
    ret ch;
}

fn parse_ident(@pstate st, str_def sd, char last) -> ast::ident {
    auto res = "";
    while (peek(st) as char != last) {
        res += str::unsafe_from_byte(next(st));
    }
    ret res;
}

fn parse_ty_data(vec[u8] data, int crate_num, uint pos, uint len,
                 str_def sd, ty::ctxt tcx) -> ty::t {
    auto st = @rec(data=data, crate=crate_num,
                   mutable pos=pos, len=len, tcx=tcx);
    auto result = parse_ty(st, sd);
    ret result;
}

fn parse_ty_or_bang(@pstate st, str_def sd) -> ty_or_bang {
    alt (peek(st) as char) {
        case ('!') { auto ignore = next(st); ret a_bang[ty::t]; }
        case (_)   { ret a_ty[ty::t](parse_ty(st, sd)); }
    }
}

fn parse_constrs(@pstate st, str_def sd) -> vec[@ast::constr] {
    let vec[@ast::constr] res = [];
    alt (peek(st) as char) {
        case (':') {
            do {
                auto ignore = next(st);
                vec::push(res, parse_constr(st, sd));
            } while (peek(st) as char == ',')
        }
        case (_) {}
    }
    ret res;
}

fn parse_constr(@pstate st, str_def sd) -> @ast::constr {
    let vec[@ast::constr_arg] args = [];
    auto sp = rec(lo=0u,hi=0u); // FIXME
    let vec[ast::ident] ids = [];
    let vec[@ast::ty] tys = [];
    let ast::path pth = respan(sp,
                               rec(idents=ids, types=tys)); // FIXME
    let ast::ident p1 = parse_ident(st, sd, '(');
    log_err("ignore=");
    log_err(p1);
    let char ignore = next(st) as char;
    assert(ignore as char == '(');
    do {
        alt (peek(st) as char) {
            case ('*') {
                auto ignore = next(st);
                args += [@respan(sp, ast::carg_base)];
            }
            case (?c) {
                log_err("c =");
                log_err(str::from_bytes([c as u8]));
                if (may_begin_ident(c)) {
                    auto id = parse_ident(st, sd, ',');
                    args += [@respan(sp, ast::carg_ident(id))];
                }
                else {
                    log_err("Lit args are unimplemented");
                    fail; // FIXME
                }
                /*
                else {
                    auto lit = parse_lit(st, sd, ',');
                    args += [respan(st.span, ast::carg_lit(lit))];
                }
                */
            }
        }
    } while (next(st) as char == ',');
    ignore = next(st) as char;
    ret @respan(sp, rec(path=pth, args=args));
}

fn parse_ty(@pstate st, str_def sd) -> ty::t {
    alt (next(st) as char) {
        case ('n') { ret ty::mk_nil(st.tcx); }
        case ('z') { ret ty::mk_bot(st.tcx); }
        case ('b') { ret ty::mk_bool(st.tcx); }
        case ('i') { ret ty::mk_int(st.tcx); }
        case ('u') { ret ty::mk_uint(st.tcx); }
        case ('l') { ret ty::mk_float(st.tcx); }
        case ('M') {
            alt (next(st) as char) {
                case ('b') { ret ty::mk_mach(st.tcx, common::ty_u8); }
                case ('w') { ret ty::mk_mach(st.tcx, common::ty_u16); }
                case ('l') { ret ty::mk_mach(st.tcx, common::ty_u32); }
                case ('d') { ret ty::mk_mach(st.tcx, common::ty_u64); }
                case ('B') { ret ty::mk_mach(st.tcx, common::ty_i8); }
                case ('W') { ret ty::mk_mach(st.tcx, common::ty_i16); }
                case ('L') { ret ty::mk_mach(st.tcx, common::ty_i32); }
                case ('D') { ret ty::mk_mach(st.tcx, common::ty_i64); }
                case ('f') { ret ty::mk_mach(st.tcx, common::ty_f32); }
                case ('F') { ret ty::mk_mach(st.tcx, common::ty_f64); }
            }
        }
        case ('c') { ret ty::mk_char(st.tcx); }
        case ('s') { ret ty::mk_str(st.tcx); }
        case ('S') { ret ty::mk_istr(st.tcx); }
        case ('t') {
            assert (next(st) as char == '[');
            auto def = parse_def(st, sd);
            let vec[ty::t] params = [];
            while (peek(st) as char != ']') {
                params += [parse_ty(st, sd)];
            }
            st.pos = st.pos + 1u;
            ret ty::mk_tag(st.tcx, def, params);
        }
        case ('p') { ret ty::mk_param(st.tcx, parse_int(st) as uint); }
        case ('@') { ret ty::mk_box(st.tcx, parse_mt(st, sd)); }
        case ('*') { ret ty::mk_ptr(st.tcx, parse_mt(st, sd)); }
        case ('V') { ret ty::mk_vec(st.tcx, parse_mt(st, sd)); }
        case ('I') { ret ty::mk_ivec(st.tcx, parse_mt(st, sd)); }
        case ('a') { ret ty::mk_task(st.tcx); }
        case ('P') { ret ty::mk_port(st.tcx, parse_ty(st, sd)); }
        case ('C') { ret ty::mk_chan(st.tcx, parse_ty(st, sd)); }
        case ('T') {
            assert (next(st) as char == '[');
            let vec[ty::mt] params = [];
            while (peek(st) as char != ']') {
                params += [parse_mt(st, sd)];
            }
            st.pos = st.pos + 1u;
            ret ty::mk_tup(st.tcx, params);
        }
        case ('R') {
            assert (next(st) as char == '[');
            let vec[ty::field] fields = [];
            while (peek(st) as char != ']') {
                auto name = "";
                while (peek(st) as char != '=') {
                    name += str::unsafe_from_byte(next(st));
                }
                st.pos = st.pos + 1u;
                fields += [rec(ident=name, mt=parse_mt(st, sd))];
            }
            st.pos = st.pos + 1u;
            ret ty::mk_rec(st.tcx, fields);
        }
        case ('F') {
            auto func = parse_ty_fn(st, sd);
            ret ty::mk_fn(st.tcx, ast::proto_fn, func._0,
                          func._1, func._2, func._3);
        }
        case ('W') {
            auto func = parse_ty_fn(st, sd);
            ret ty::mk_fn(st.tcx, ast::proto_iter, func._0,
                          func._1, func._2, func._3);
        }
        case ('N') {
            auto abi;
            alt (next(st) as char) {
                case ('r') { abi = ast::native_abi_rust; }
                case ('i') { abi = ast::native_abi_rust_intrinsic; }
                case ('c') { abi = ast::native_abi_cdecl; }
                case ('l') { abi = ast::native_abi_llvm; }
            }
            auto func = parse_ty_fn(st, sd);
            ret ty::mk_native_fn(st.tcx,abi,func._0,func._1);
        }
        case ('O') {
            assert (next(st) as char == '[');
            let vec[ty::method] methods = [];
            while (peek(st) as char != ']') {
                auto proto;
                alt (next(st) as char) {
                    case ('W') {proto = ast::proto_iter;}
                    case ('F') {proto = ast::proto_fn;}
                }
                auto name = "";
                while (peek(st) as char != '[') {
                    name += str::unsafe_from_byte(next(st));
                }
                auto func = parse_ty_fn(st, sd);
                methods += [rec(proto=proto,
                                ident=name,
                                inputs=func._0,
                                output=func._1,
                                cf=func._2,
                                constrs=func._3)];
            }
            st.pos += 1u;
            ret ty::mk_obj(st.tcx, methods);
        }
        case ('X') { ret ty::mk_var(st.tcx, parse_int(st)); }
        case ('E') { ret ty::mk_native(st.tcx); }
        case ('Y') { ret ty::mk_type(st.tcx); }
        case ('#') {
            auto pos = parse_hex(st);
            assert (next(st) as char == ':');
            auto len = parse_hex(st);
            assert (next(st) as char == '#');
            alt (st.tcx.rcache.find(tup(st.crate,pos,len))) {
                case (some(?tt)) { ret tt; }
                case (none) {
                    auto ps = @rec(pos=pos, len=len with *st);
                    auto tt = parse_ty(ps, sd);
                    st.tcx.rcache.insert(tup(st.crate,pos,len), tt);
                    ret tt;
                }
            }
        }
        case (?c) {
            log_err "unexpected char in type string: ";
            log_err c;
            fail;
        }
    }
}

fn parse_mt(@pstate st, str_def sd) -> ty::mt {
    auto mut;
    alt (peek(st) as char) {
        case ('m') {next(st); mut = ast::mut;}
        case ('?') {next(st); mut = ast::maybe_mut;}
        case (_)   {mut=ast::imm;}
    }
    ret rec(ty=parse_ty(st, sd), mut=mut);
}

fn parse_def(@pstate st, str_def sd) -> ast::def_id {
    auto def = "";
    while (peek(st) as char != '|') {
        def += str::unsafe_from_byte(next(st));
    }
    st.pos = st.pos + 1u;
    ret sd(def);
}

fn parse_int(@pstate st) -> int {
    auto n = 0;
    while (true) {
        auto cur = peek(st) as char;
        if (cur < '0' || cur > '9') {break;}
        st.pos = st.pos + 1u;
        n *= 10;
        n += (cur as int) - ('0' as int);
    }
    ret n;
}

fn parse_hex(@pstate st) -> uint {
    auto n = 0u;
    while (true) {
        auto cur = peek(st) as char;
        if ((cur < '0' || cur > '9') &&
            (cur < 'a' || cur > 'f')) {break;}
        st.pos = st.pos + 1u;
        n *= 16u;
        if ('0' <= cur && cur <= '9') {
            n += (cur as uint) - ('0' as uint);
        } else {
            n += (10u + (cur as uint) - ('a' as uint));
        }
    }
    ret n;
}

fn parse_ty_fn(@pstate st, str_def sd) -> tup(vec[ty::arg], ty::t,
                                              ast::controlflow,
                                              vec[@ast::constr]) {
    assert (next(st) as char == '[');
    let vec[ty::arg] inputs = [];
    while (peek(st) as char != ']') {
        auto mode = ty::mo_val;
        if (peek(st) as char == '&') {
            mode = ty::mo_alias;
            st.pos = st.pos + 1u;
        }
        inputs += [rec(mode=mode, ty=parse_ty(st, sd))];
    }
    st.pos = st.pos + 1u;
    auto cs  = parse_constrs(st, sd);
    auto res = parse_ty_or_bang(st, sd);
    alt (res) {
        case (a_bang) {
            ret tup(inputs, ty::mk_bot(st.tcx), ast::noreturn, cs);
        }
        case (a_ty(?t)) {
            ret tup(inputs, t, ast::return, cs);
        }
    }
    
}


// Rust metadata parsing

fn parse_def_id(vec[u8] buf) -> ast::def_id {
    auto colon_idx = 0u;
    auto len = vec::len[u8](buf);
    while (colon_idx < len && buf.(colon_idx) != (':' as u8)) {
        colon_idx += 1u;
    }
    if (colon_idx == len) {
        log_err "didn't find ':' when parsing def id";
        fail;
    }

    auto crate_part = vec::slice[u8](buf, 0u, colon_idx);
    auto def_part = vec::slice[u8](buf, colon_idx + 1u, len);
    auto crate_num = uint::parse_buf(crate_part, 10u) as int;
    auto def_num = uint::parse_buf(def_part, 10u) as int;
    ret tup(crate_num, def_num);
}

fn lookup_hash(&ebml::doc d, fn(vec[u8]) -> bool eq_fn, uint hash)
    -> vec[ebml::doc] {
    auto index = ebml::get_doc(d, metadata::tag_index);
    auto table = ebml::get_doc(index, metadata::tag_index_table);

    auto hash_pos = table.start + (hash % 256u) * 4u;
    auto pos = ebml::be_uint_from_bytes(d.data, hash_pos, 4u);
    auto bucket = ebml::doc_at(d.data, pos);
    // Awkward logic because we can't ret from foreach yet
    let vec[ebml::doc] result = [];
    auto belt = metadata::tag_index_buckets_bucket_elt;
    for each (ebml::doc elt in ebml::tagged_docs(bucket, belt)) {
        auto pos = ebml::be_uint_from_bytes(elt.data, elt.start, 4u);
        if (eq_fn(vec::slice[u8](elt.data, elt.start+4u, elt.end))) {
            vec::push(result, ebml::doc_at(d.data, pos));
        }
    }
    ret result;
}

// Given a path and serialized crate metadata, returns the ID of the
// definition the path refers to.
fn resolve_path(vec[ast::ident] path, vec[u8] data) -> vec[ast::def_id] {
    fn eq_item(vec[u8] data, str s) -> bool {
        ret str::eq(str::unsafe_from_bytes(data), s);
    }
    auto s = str::connect(path, "::");
    auto md = ebml::new_doc(data);
    auto paths = ebml::get_doc(md, metadata::tag_paths);
    auto eqer = bind eq_item(_, s);
    let vec[ast::def_id] result = [];
    for (ebml::doc doc in lookup_hash(paths, eqer, metadata::hash_path(s))) {
        auto did_doc = ebml::get_doc(doc, metadata::tag_def_id);
        vec::push(result, parse_def_id(ebml::doc_data(did_doc)));
    }
    ret result;
}

fn maybe_find_item(int item_id, &ebml::doc items) -> option::t[ebml::doc] {
    fn eq_item(vec[u8] bytes, int item_id) -> bool {
        ret ebml::be_uint_from_bytes(bytes, 0u, 4u) as int == item_id;
    }
    auto eqer = bind eq_item(_, item_id);
    auto found = lookup_hash(items, eqer, metadata::hash_def_num(item_id));
    if (vec::len(found) == 0u) {
        ret option::none[ebml::doc];
    } else {
        ret option::some[ebml::doc](found.(0));
    }
}

fn find_item(int item_id, &ebml::doc items) -> ebml::doc {
    ret option::get(maybe_find_item(item_id, items));
}

// Looks up an item in the given metadata and returns an ebml doc pointing
// to the item data.
fn lookup_item(int item_id, vec[u8] data) -> ebml::doc {
    auto items = ebml::get_doc(ebml::new_doc(data), metadata::tag_items);
    ret find_item(item_id, items);
}

fn item_kind(&ebml::doc item) -> u8 {
    auto kind = ebml::get_doc(item, metadata::tag_items_data_item_kind);
    ret ebml::doc_as_uint(kind) as u8;
}

fn item_symbol(&ebml::doc item) -> str {
    auto sym = ebml::get_doc(item, metadata::tag_items_data_item_symbol);
    ret str::unsafe_from_bytes(ebml::doc_data(sym));
}

fn variant_tag_id(&ebml::doc d) -> ast::def_id {
    auto tagdoc = ebml::get_doc(d, metadata::tag_items_data_item_tag_id);
    ret parse_def_id(ebml::doc_data(tagdoc));
}

fn item_type(&ebml::doc item, int this_cnum, ty::ctxt tcx) -> ty::t {
    fn parse_external_def_id(int this_cnum, str s) -> ast::def_id {
        // FIXME: This is completely wrong when linking against a crate
        // that, in turn, links against another crate. We need a mapping
        // from crate ID to crate "meta" attributes as part of the crate
        // metadata:
        auto buf = str::bytes(s);
        auto external_def_id = parse_def_id(buf);
        ret tup(this_cnum, external_def_id._1);
    }

    auto tp = ebml::get_doc(item, metadata::tag_items_data_item_type);
    auto s = str::unsafe_from_bytes(ebml::doc_data(tp));
    ret parse_ty_data(item.data, this_cnum, tp.start, tp.end - tp.start,
                      bind parse_external_def_id(this_cnum, _), tcx);
}

fn item_ty_param_count(&ebml::doc item, int this_cnum) -> uint {
    let uint ty_param_count = 0u;
    auto tp = metadata::tag_items_data_item_ty_param_count;
    for each (ebml::doc p in ebml::tagged_docs(item, tp)) {
        ty_param_count = ebml::vint_at(ebml::doc_data(p), 0u)._0;
    }
    ret ty_param_count;
}

fn tag_variant_ids(&ebml::doc item, int this_cnum) -> vec[ast::def_id] {
    let vec[ast::def_id] ids = [];
    auto v = metadata::tag_items_data_item_variant;
    for each (ebml::doc p in ebml::tagged_docs(item, v)) {
        auto ext = parse_def_id(ebml::doc_data(p));
        vec::push[ast::def_id](ids, tup(this_cnum, ext._1));
    }
    ret ids;
}

fn get_metadata_section(str filename) -> option::t[vec[u8]] {
    auto mb = llvm::LLVMRustCreateMemoryBufferWithContentsOfFile
        (str::buf(filename));
    if (mb as int == 0) {ret option::none[vec[u8]];}
    auto of = mk_object_file(mb);
    auto si = mk_section_iter(of.llof);
    while (llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False) {
        auto name_buf = llvm::LLVMGetSectionName(si.llsi);
        auto name = str::str_from_cstr(name_buf);
        if (str::eq(name, x86::get_meta_sect_name())) {
            auto cbuf = llvm::LLVMGetSectionContents(si.llsi);
            auto csz = llvm::LLVMGetSectionSize(si.llsi);
            auto cvbuf = cbuf as vec::vbuf;
            ret option::some[vec[u8]](vec::vec_from_vbuf[u8](cvbuf, csz));
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none[vec[u8]];
}


fn metadata_matches(&vec[u8] data,
                    &vec[@ast::meta_item] metas) -> bool {
    ret true;
}

fn find_library_crate(&session::session sess,
                      &ast::ident ident,
                      &vec[@ast::meta_item] metas,
                      &vec[str] library_search_paths)
    -> option::t[tup(str, vec[u8])] {

    let str crate_name = ident;
    for (@ast::meta_item mi in metas) {
        if (mi.node.key == "name") {
            crate_name = mi.node.value;
            break;
        }
    }
    auto nn = parser::default_native_lib_naming(sess);
    let str prefix = nn.prefix + crate_name;

    // FIXME: we could probably use a 'glob' function in std::fs but it will
    // be much easier to write once the unsafe module knows more about FFI
    // tricks. Currently the glob(3) interface is a bit more than we can
    // stomach from here, and writing a C++ wrapper is more work than just
    // manually filtering fs::list_dir here.

    for (str library_search_path in library_search_paths) {

        for (str path in fs::list_dir(library_search_path)) {

            let str f = fs::basename(path);
            if (! (str::starts_with(f, prefix) &&
                   str::ends_with(f, nn.suffix))) {
                log #fmt("skipping %s, doesn't look like %s*%s",
                         path, prefix, nn.suffix);
                cont;
            }

            alt (get_metadata_section(path)) {
                case (option::some(?cvec)) {
                    if (!metadata_matches(cvec, metas)) {
                        log #fmt("skipping %s, metadata doesn't match", path);
                        cont;
                    }
                    log #fmt("found %s with matching metadata", path);
                    ret some(tup(path, cvec));
                }
                case (_) {}
            }
        }
    }
    ret none;
}

fn load_library_crate(&session::session sess,
                      &int cnum,
                      &ast::ident ident,
                      &vec[@ast::meta_item] metas,
                      &vec[str] library_search_paths) {
    alt (find_library_crate(sess, ident, metas, library_search_paths)) {
        case (some(?t)) {
            sess.set_external_crate(cnum, rec(name=ident,
                                              data=t._1));
            ret;
        }
        case (_) {}
    }
    log_err #fmt("can't find crate for '%s'", ident);
    fail;
}

type env = @rec(
    session::session sess,
    resolve::crate_map crate_map,
    @hashmap[str, int] crate_cache,
    vec[str] library_search_paths,
    mutable int next_crate_num
);

fn visit_view_item(env e, &@ast::view_item i) {
    alt (i.node) {
        case (ast::view_item_use(?ident, ?meta_items, ?id, ?ann)) {
            auto cnum;
            if (!e.crate_cache.contains_key(ident)) {
                cnum = e.next_crate_num;
                load_library_crate(e.sess, cnum, ident, meta_items,
                                   e.library_search_paths);
                e.crate_cache.insert(ident, e.next_crate_num);
                e.next_crate_num += 1;
            } else {
                cnum = e.crate_cache.get(ident);
            }
            e.crate_map.insert(ann.id, cnum);
        }
        case (_) { }
    }
}


// Reads external crates referenced by "use" directives.
fn read_crates(session::session sess,
               resolve::crate_map crate_map,
               &ast::crate crate) {
    auto e = @rec(
        sess=sess,
        crate_map=crate_map,
        crate_cache=@common::new_str_hash[int](),
        library_search_paths=sess.get_opts().library_search_paths,
        mutable next_crate_num=1
    );

    auto v = rec(visit_view_item_pre=bind visit_view_item(e, _)
                 with walk::default_visitor());
    walk::walk_crate(v, crate);
}


fn kind_has_type_params(u8 kind_ch) -> bool {
    ret alt (kind_ch as char) {
        case ('c') { false } case ('f') { true  } case ('F') { true  }
        case ('y') { true  } case ('o') { true  } case ('t') { true  }
        case ('T') { false } case ('m') { false } case ('n') { false }
        case ('v') { true  }
    };
}

// Crate metadata queries

fn lookup_defs(session::session sess, int cnum, vec[ast::ident] path)
    -> vec[ast::def] {
    auto data = sess.get_external_crate(cnum).data;

    ret vec::map(bind lookup_def(cnum, data, _),
                  resolve_path(path, data));
}

// FIXME doesn't yet handle re-exported externals
fn lookup_def(int cnum, vec[u8] data, &ast::def_id did) -> ast::def {
    auto item = lookup_item(did._1, data);
    auto kind_ch = item_kind(item);

    did = tup(cnum, did._1);

    auto def = alt (kind_ch as char) {
        case ('c') { ast::def_const(did) }
        case ('f') { ast::def_fn(did) }
        case ('F') { ast::def_native_fn(did) }
        case ('y') { ast::def_ty(did) }
        case ('o') { ast::def_obj(did) }
        case ('T') { ast::def_native_ty(did) }
        // We treat references to tags as references to types.
        case ('t') { ast::def_ty(did) }
        case ('m') { ast::def_mod(did) }
        case ('n') { ast::def_native_mod(did) }
        case ('v') {
            auto tid = variant_tag_id(item);
            tid = tup(cnum, tid._1);
            ast::def_variant(tid, did)
        }
    };
    ret def;
}

fn get_type(ty::ctxt tcx, ast::def_id def) -> ty::ty_param_count_and_ty {
    auto external_crate_id = def._0;
    auto data = tcx.sess.get_external_crate(external_crate_id).data;
    auto item = lookup_item(def._1, data);
    auto t = item_type(item, external_crate_id, tcx);

    auto tp_count;
    auto kind_ch = item_kind(item);
    auto has_ty_params = kind_has_type_params(kind_ch);
    if (has_ty_params) {
        tp_count = item_ty_param_count(item, external_crate_id);
    } else {
        tp_count = 0u;
    }

    ret tup(tp_count, t);
}

fn get_symbol(session::session sess, ast::def_id def) -> str {
    auto external_crate_id = def._0;
    auto data = sess.get_external_crate(external_crate_id).data;
    ret item_symbol(lookup_item(def._1, data));
}

fn get_tag_variants(ty::ctxt tcx, ast::def_id def)
    -> vec[ty::variant_info] {
    auto external_crate_id = def._0;
    auto data = tcx.sess.get_external_crate(external_crate_id).data;
    auto items = ebml::get_doc(ebml::new_doc(data), metadata::tag_items);
    auto item = find_item(def._1, items);

    let vec[ty::variant_info] infos = [];
    auto variant_ids = tag_variant_ids(item, external_crate_id);
    for (ast::def_id did in variant_ids) {
        auto item = find_item(did._1, items);
        auto ctor_ty = item_type(item, external_crate_id, tcx);
        let vec[ty::t] arg_tys = [];
        alt (ty::struct(tcx, ctor_ty)) {
            case (ty::ty_fn(_, ?args, _, _, _)) {
                for (ty::arg a in args) {
                    arg_tys += [a.ty];
                }
            }
            case (_) {
                // Nullary tag variant.
            }
        }
        infos += [rec(args=arg_tys, ctor_ty=ctor_ty, id=did)];
    }

    ret infos;
}

fn list_file_metadata(str path, io::writer out) {
    alt (get_metadata_section(path)) {
        case (option::some(?bytes)) {
            list_crate_metadata(bytes, out);
        }
        case (option::none) {
            out.write_str("Could not find metadata in " + path + ".\n");
        }
    }
}

fn read_path(&ebml::doc d) -> tup(str, uint) {
    auto desc = ebml::doc_data(d);
    auto pos = ebml::be_uint_from_bytes(desc, 0u, 4u);
    auto pathbytes = vec::slice[u8](desc, 4u, vec::len[u8](desc));
    auto path = str::unsafe_from_bytes(pathbytes);
    ret tup(path, pos);
}

fn list_crate_metadata(vec[u8] bytes, io::writer out) {
    auto md = ebml::new_doc(bytes);
    auto paths = ebml::get_doc(md, metadata::tag_paths);
    auto items = ebml::get_doc(md, metadata::tag_items);
    auto index = ebml::get_doc(paths, metadata::tag_index);
    auto bs = ebml::get_doc(index, metadata::tag_index_buckets);
    for each (ebml::doc bucket in
              ebml::tagged_docs(bs, metadata::tag_index_buckets_bucket)) {
        auto et = metadata::tag_index_buckets_bucket_elt;
        for each (ebml::doc elt in ebml::tagged_docs(bucket, et)) {
            auto data = read_path(elt);
            auto def = ebml::doc_at(bytes, data._1);
            auto did_doc = ebml::get_doc(def, metadata::tag_def_id);
            auto did = parse_def_id(ebml::doc_data(did_doc));
            out.write_str(#fmt("%s (%s)\n", data._0,
                               describe_def(items, did)));
        }
    }
}

fn describe_def(&ebml::doc items, ast::def_id id) -> str {
    if (id._0 != 0) { ret "external"; }
    ret item_kind_to_str(item_kind(find_item(id._1, items)));
}

fn item_kind_to_str(u8 kind) -> str {
    alt (kind as char) {
        case ('c') {ret "const";}
        case ('f') {ret "fn";}
        case ('F') {ret "native fn";}
        case ('y') {ret "type";}
        case ('o') {ret "obj";}
        case ('T') {ret "native type";}
        case ('t') {ret "type";}
        case ('m') {ret "mod";}
        case ('n') {ret "native mod";}
        case ('v') {ret "tag";}
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
