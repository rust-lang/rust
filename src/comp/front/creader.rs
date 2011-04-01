// -*- rust -*-

import driver.session;
import front.ast;
import lib.llvm.False;
import lib.llvm.llvm;
import lib.llvm.mk_object_file;
import lib.llvm.mk_section_iter;
import middle.fold;
import middle.metadata;
import middle.ty;
import back.x86;
import util.common;
import util.common.span;

import std._str;
import std._uint;
import std._vec;
import std.ebml;
import std.fs;
import std.io;
import std.option;
import std.option.none;
import std.option.some;
import std.os;
import std.map.hashmap;

// TODO: map to a real type here.
type env = @rec(
    session.session sess,
    @hashmap[str, int] crate_cache,
    vec[str] library_search_paths,
    mutable int next_crate_num
);

tag resolve_result {
    rr_ok(ast.def_id);
    rr_not_found(ast.ident);
}

// Type decoding

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// (The second has to be authed pure.) Extra parameters are for converting
// to/from def_ids in the string rep. Whatever format you choose should not
// contain pipe characters.

// Callback to translate defs to strs or back.
type str_def = fn(str) -> ast.def_id;

type pstate = rec(str rep, mutable uint pos, uint len);

fn peek(@pstate st) -> u8 {
    if (st.pos < st.len) {ret st.rep.(st.pos) as u8;}
    else {ret ' ' as u8;}
}
impure fn next(@pstate st) -> u8 { // ?? somehow not recognized as impure
    if (st.pos >= st.len) {fail;}
    auto ch = st.rep.(st.pos);
    st.pos = st.pos + 1u;
    ret ch as u8;
}

impure fn parse_ty_str(str rep, str_def sd) -> @ty.t {
    auto len = _str.byte_len(rep);
    auto st = @rec(rep=rep, mutable pos=0u, len=len);
    auto result = parse_ty(st, sd);
    if (st.pos != len) {
        log "parse_ty_str: incomplete parse, stopped at byte "
            + _uint.to_str(st.pos, 10u) + " of "
            + _uint.to_str(len, 10u) + " in str '" + rep + "'";
    }
    check(st.pos == len);
    ret result;
}

impure fn parse_ty(@pstate st, str_def sd) -> @ty.t {
    ret @rec(struct=parse_sty(st, sd),
             cname=option.none[str]);
}

impure fn parse_mt(@pstate st, str_def sd) -> ty.mt {
    auto mut;
    alt (peek(st) as char) {
        case ('m') {next(st); mut = ast.mut;}
        case ('?') {next(st); mut = ast.maybe_mut;}
        case (_)   {mut=ast.imm;}
    }
    ret rec(ty=parse_ty(st, sd), mut=mut);
}

impure fn parse_def(@pstate st, str_def sd) -> ast.def_id {
    auto def = "";
    while (peek(st) as char != '|') {
        def += _str.unsafe_from_byte(next(st));
    }
    st.pos = st.pos + 1u;
    ret sd(def);
}

impure fn parse_sty(@pstate st, str_def sd) -> ty.sty {
    alt (next(st) as char) {
        case ('n') {ret ty.ty_nil;}
        case ('b') {ret ty.ty_bool;}
        case ('i') {ret ty.ty_int;}
        case ('u') {ret ty.ty_uint;}
        case ('l') {ret ty.ty_float;}
        case ('M') {
            alt (next(st) as char) {
                case ('b') {ret ty.ty_machine(common.ty_u8);}
                case ('w') {ret ty.ty_machine(common.ty_u16);}
                case ('l') {ret ty.ty_machine(common.ty_u32);}
                case ('d') {ret ty.ty_machine(common.ty_u64);}
                case ('B') {ret ty.ty_machine(common.ty_i8);}
                case ('W') {ret ty.ty_machine(common.ty_i16);}
                case ('L') {ret ty.ty_machine(common.ty_i32);}
                case ('D') {ret ty.ty_machine(common.ty_i64);}
                case ('f') {ret ty.ty_machine(common.ty_f32);}
                case ('F') {ret ty.ty_machine(common.ty_f64);}
            }
        }
        case ('c') {ret ty.ty_char;}
        case ('s') {ret ty.ty_str;}
        case ('t') {
            check(next(st) as char == '[');
            auto def = parse_def(st, sd);
            let vec[@ty.t] params = vec();
            while (peek(st) as char != ']') {
                params += vec(parse_ty(st, sd));
            }
            st.pos = st.pos + 1u;
            ret ty.ty_tag(def, params);
        }
        case ('p') {ret ty.ty_param(parse_def(st, sd));}
        case ('@') {ret ty.ty_box(parse_mt(st, sd));}
        case ('V') {ret ty.ty_vec(parse_mt(st, sd));}
        case ('P') {ret ty.ty_port(parse_ty(st, sd));}
        case ('C') {ret ty.ty_chan(parse_ty(st, sd));}
        case ('T') {
            check(next(st) as char == '[');
            let vec[ty.mt] params = vec();
            while (peek(st) as char != ']') {
                params += vec(parse_mt(st, sd));
            }
            st.pos = st.pos + 1u;
            ret ty.ty_tup(params);
        }
        case ('R') {
            check(next(st) as char == '[');
            let vec[ty.field] fields = vec();
            while (peek(st) as char != ']') {
                auto name = "";
                while (peek(st) as char != '=') {
                    name += _str.unsafe_from_byte(next(st));
                }
                st.pos = st.pos + 1u;
                fields += vec(rec(ident=name, mt=parse_mt(st, sd)));
            }
            st.pos = st.pos + 1u;
            ret ty.ty_rec(fields);
        }
        case ('F') {
            auto func = parse_ty_fn(st, sd);
            ret ty.ty_fn(ast.proto_fn, func._0, func._1);
        }
        case ('W') {
            auto func = parse_ty_fn(st, sd);
            ret ty.ty_fn(ast.proto_iter, func._0, func._1);
        }
        case ('N') {
            auto abi;
            alt (next(st) as char) {
                case ('r') {abi = ast.native_abi_rust;}
                case ('c') {abi = ast.native_abi_cdecl;}
                case ('l') {abi = ast.native_abi_llvm;}
            }
            auto func = parse_ty_fn(st, sd);
            ret ty.ty_native_fn(abi,func._0,func._1);
        }
        case ('O') {
            check(next(st) as char == '[');
            let vec[ty.method] methods = vec();
            while (peek(st) as char != ']') {
                auto proto;
                alt (next(st) as char) {
                    case ('W') {proto = ast.proto_iter;}
                    case ('F') {proto = ast.proto_fn;}
                }
                auto name = "";
                while (peek(st) as char != '[') {
                    name += _str.unsafe_from_byte(next(st));
                }
                auto func = parse_ty_fn(st, sd);
                methods += vec(rec(proto=proto,
                                   ident=name,
                                   inputs=func._0,
                                   output=func._1));
            }
            st.pos += 1u;
            ret ty.ty_obj(methods);
        }
        case ('X') {ret ty.ty_var(parse_int(st));}
        case ('E') {ret ty.ty_native;}
        case ('Y') {ret ty.ty_type;}
    }
}

impure fn parse_int(@pstate st) -> int {
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

impure fn parse_ty_fn(@pstate st, str_def sd) -> tup(vec[ty.arg], @ty.t) {
    check(next(st) as char == '[');
    let vec[ty.arg] inputs = vec();
    while (peek(st) as char != ']') {
        auto mode = ast.val;
        if (peek(st) as char == '&') {
            mode = ast.alias;
            st.pos = st.pos + 1u;
        }
        inputs += vec(rec(mode=mode, ty=parse_ty(st, sd)));
    }
    st.pos = st.pos + 1u;
    ret tup(inputs, parse_ty(st, sd));
}


// Rust metadata parsing

fn parse_def_id(vec[u8] buf) -> ast.def_id {
    auto colon_idx = 0u;
    auto len = _vec.len[u8](buf);
    while (colon_idx < len && buf.(colon_idx) != (':' as u8)) {
        colon_idx += 1u;
    }
    if (colon_idx == len) {
        log "didn't find ':' when parsing def id";
        fail;
    }

    auto crate_part = _vec.slice[u8](buf, 0u, colon_idx);
    auto def_part = _vec.slice[u8](buf, colon_idx + 1u, len);
    auto crate_num = _uint.parse_buf(crate_part, 10u) as int;
    auto def_num = _uint.parse_buf(def_part, 10u) as int;
    ret tup(crate_num, def_num);
}

impure fn lookup_hash_entry(&ebml.reader ebml_r,
                            impure fn(&ebml.reader) -> bool eq_fn,
                            uint hash) -> bool {
    ebml.move_to_child_with_id(ebml_r, metadata.tag_index);
    ebml.move_to_child_with_id(ebml_r, metadata.tag_index_table);
    ebml.move_to_first_child(ebml_r);

    // Move to the bucket.
    auto bucket_index = hash % 256u;
    auto buf_reader = ebml_r.reader.get_buf_reader();
    buf_reader.seek((bucket_index * 4u) as int, io.seek_cur);
    auto bucket_pos = ebml_r.reader.read_be_uint(4u);
    ebml.reset_reader(ebml_r, bucket_pos);

    // Search to find the item ID in the bucket.
    check (ebml.peek(ebml_r).id == metadata.tag_index_buckets_bucket);
    ebml.move_to_first_child(ebml_r);
    while (ebml.bytes_left(ebml_r) > 0u) {
        if (ebml.peek(ebml_r).id == metadata.tag_index_buckets_bucket_elt) {
            ebml.move_to_first_child(ebml_r);
            auto pos = ebml_r.reader.read_be_uint(4u);
            if (eq_fn(ebml_r)) {
                // Found the item. Move to its data and return.
                ebml.reset_reader(ebml_r, pos);
                ebml.move_to_first_child(ebml_r);
                ret true;
            }
            ebml.move_to_parent(ebml_r);
        }
        ebml.move_to_next_sibling(ebml_r);
    }

    ret false;
}

// Given a path and serialized crate metadata, returns the ID of the
// definition the path refers to.
impure fn resolve_path(vec[ast.ident] path, vec[u8] data) -> resolve_result {
    impure fn eq_item(&ebml.reader ebml_r, str s) -> bool {
        auto this_str = _str.unsafe_from_bytes(ebml.read_data(ebml_r));
        ret _str.eq(this_str, s);
    }

    auto s = _str.connect(path, ".");

    auto io_r = io.new_reader_(io.new_byte_buf_reader(data));
    auto ebml_r = ebml.create_reader(io_r);
    ebml.move_to_sibling_with_id(ebml_r, metadata.tag_paths);

    auto eqer = bind eq_item(_, s);
    auto hash = metadata.hash_path(s);
    if (!lookup_hash_entry(ebml_r, eqer, hash)) {
        ret rr_not_found(s);
    }

    ebml.move_to_sibling_with_id(ebml_r, metadata.tag_def_id);
    ebml.move_to_first_child(ebml_r);
    auto did_data = ebml.read_data(ebml_r);
    ebml.move_to_parent(ebml_r);
    auto did = parse_def_id(did_data);
    ret rr_ok(did);
}

impure fn move_to_item(&ebml.reader ebml_r, int item_id) {
    impure fn eq_item(&ebml.reader ebml_r, int item_id) -> bool {
        ret (ebml_r.reader.read_be_uint(4u) as int) == item_id;
    }

    auto eqer = bind eq_item(_, item_id);
    auto hash = metadata.hash_def_num(item_id);
    ebml.move_to_sibling_with_id(ebml_r, metadata.tag_items);
    lookup_hash_entry(ebml_r, eqer, hash);
}

// Looks up an item in the given metadata and returns an EBML reader pointing
// to the item data.
impure fn lookup_item(int item_id, vec[u8] data) -> ebml.reader {
    auto io_r = io.new_reader_(io.new_byte_buf_reader(data));
    auto ebml_r = ebml.create_reader(io_r);
    move_to_item(ebml_r, item_id);
    ret ebml_r;
}

impure fn get_item_generic[T](&ebml.reader ebml_r, uint tag_id,
        impure fn(vec[u8] buf) -> T converter) -> T {
    while (ebml.bytes_left(ebml_r) > 0u) {
        auto ebml_tag = ebml.peek(ebml_r);
        if (ebml_tag.id == tag_id) {
            ebml.move_to_first_child(ebml_r);
            auto result = converter(ebml.read_data(ebml_r));

            // Be kind, rewind.
            ebml.move_to_parent(ebml_r);
            ebml.move_to_parent(ebml_r);
            ebml.move_to_first_child(ebml_r);

            ret result;
        }
        ebml.move_to_next_sibling(ebml_r);
    }

    log #fmt("get_item_generic(): tag %u not found", tag_id);
    fail;
}

impure fn get_item_kind(&ebml.reader ebml_r) -> u8 {
    impure fn converter(vec[u8] data) -> u8 {
        auto x = @mutable 3;
        *x = 5;
        ret data.(0);
    }
    auto f = converter;
    ret get_item_generic[u8](ebml_r, metadata.tag_items_data_item_kind, f);
}

impure fn get_item_symbol(&ebml.reader ebml_r) -> str {
    impure fn converter(vec[u8] data) -> str {
        auto x = @mutable 3;
        *x = 5;
        ret _str.unsafe_from_bytes(data);
    }
    auto f = converter;
    ret get_item_generic[str](ebml_r, metadata.tag_items_data_item_symbol, f);
}

// FIXME: This is a *terrible* botch.
impure fn impure_parse_def_id(vec[u8] data) -> ast.def_id {
    auto x = @mutable 3;
    *x = 5;
    ret parse_def_id(data);
}

impure fn get_variant_tag_id(&ebml.reader ebml_r) -> ast.def_id {
    auto f = impure_parse_def_id;
    ret get_item_generic[ast.def_id](ebml_r,
                                     metadata.tag_items_data_item_tag_id, f);
}

impure fn get_item_type(&ebml.reader ebml_r, int this_cnum) -> @ty.t {
    impure fn converter(int this_cnum, vec[u8] data) -> @ty.t {
        fn parse_external_def_id(int this_cnum, str s) -> ast.def_id {
            // FIXME: This is completely wrong when linking against a crate
            // that, in turn, links against another crate. We need a mapping
            // from crate ID to crate "meta" attributes as part of the crate
            // metadata.
            auto buf = _str.bytes(s);
            auto external_def_id = parse_def_id(buf);
            ret tup(this_cnum, external_def_id._1);
        }
        auto s = _str.unsafe_from_bytes(data);
        ret parse_ty_str(s, bind parse_external_def_id(this_cnum, _));
    }
    auto f = bind converter(this_cnum, _);
    ret get_item_generic[@ty.t](ebml_r, metadata.tag_items_data_item_type, f);
}

impure fn get_item_ty_params(&ebml.reader ebml_r, int this_cnum)
        -> vec[ast.def_id] {
    let vec[ast.def_id] tps = vec();
    while (ebml.bytes_left(ebml_r) > 0u) {
        auto ebml_tag = ebml.peek(ebml_r);
        if (ebml_tag.id == metadata.tag_items_data_item_ty_param) {
            ebml.move_to_first_child(ebml_r);

            auto data = ebml.read_data(ebml_r);
            auto external_def_id = parse_def_id(data);
            tps += vec(tup(this_cnum, external_def_id._1));

            ebml.move_to_parent(ebml_r);
        }
        ebml.move_to_next_sibling(ebml_r);
    }

    // Be kind, rewind.
    ebml.move_to_parent(ebml_r);
    ebml.move_to_first_child(ebml_r);

    ret tps;
}


fn load_crate(session.session sess,
              int cnum,
              ast.ident ident,
              vec[str] library_search_paths) {
    auto filename = parser.default_native_name(sess, ident);
    for (str library_search_path in library_search_paths) {
        auto path = fs.connect(library_search_path, filename);
        auto pbuf = _str.buf(path);
        auto mb = llvm.LLVMRustCreateMemoryBufferWithContentsOfFile(pbuf);
        if (mb as int != 0) {
            auto of = mk_object_file(mb);
            auto si = mk_section_iter(of.llof);
            while (llvm.LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) ==
                    False) {
                auto name_buf = llvm.LLVMGetSectionName(si.llsi);
                auto name = _str.str_from_cstr(name_buf);
                if (_str.eq(name, x86.get_meta_sect_name())) {
                    auto cbuf = llvm.LLVMGetSectionContents(si.llsi);
                    auto csz = llvm.LLVMGetSectionSize(si.llsi);
                    auto cvbuf = cbuf as _vec.vbuf;
                    auto cvec = _vec.vec_from_vbuf[u8](cvbuf, csz);
                    sess.set_external_crate(cnum, cvec);
                    ret;
                }
                llvm.LLVMMoveToNextSection(si.llsi);
            }
        }
    }

    log #fmt("can't open crate '%s' (looked for '%s' in lib search paths)",
        ident, filename);
    fail;
}

fn fold_view_item_use(&env e, &span sp, ast.ident ident,
        vec[@ast.meta_item] meta_items, ast.def_id id, option.t[int] cnum_opt)
        -> @ast.view_item {
    auto cnum;
    if (!e.crate_cache.contains_key(ident)) {
        cnum = e.next_crate_num;
        load_crate(e.sess, cnum, ident, e.library_search_paths);
        e.crate_cache.insert(ident, e.next_crate_num);
        e.next_crate_num += 1;
    } else {
        cnum = e.crate_cache.get(ident);
    }

    auto viu = ast.view_item_use(ident, meta_items, id, some[int](cnum));
    ret @fold.respan[ast.view_item_](sp, viu);
}

// Reads external crates referenced by "use" directives.
fn read_crates(session.session sess,
               @ast.crate crate,
               vec[str] library_search_paths) -> @ast.crate {
    auto e = @rec(
        sess=sess,
        crate_cache=@common.new_str_hash[int](),
        library_search_paths=library_search_paths,
        mutable next_crate_num=1
    );

    auto f = fold_view_item_use;
    auto fld = @rec(fold_view_item_use=f with *fold.new_identity_fold[env]());
    ret fold.fold_crate[env](e, fld, crate);
}


fn kind_has_type_params(u8 kind_ch) -> bool {
    // FIXME: It'd be great if we had u8 char literals.
    if (kind_ch == ('c' as u8))      { ret false; }
    else if (kind_ch == ('f' as u8)) { ret true;  }
    else if (kind_ch == ('F' as u8)) { ret true;  }
    else if (kind_ch == ('y' as u8)) { ret true;  }
    else if (kind_ch == ('o' as u8)) { ret true;  }
    else if (kind_ch == ('t' as u8)) { ret true;  }
    else if (kind_ch == ('m' as u8)) { ret false; }
    else if (kind_ch == ('n' as u8)) { ret false; }
    else if (kind_ch == ('v' as u8)) { ret true;  }
    else {
        log #fmt("kind_has_type_params(): unknown kind char: %d",
                 kind_ch as int);
        fail;
    }
}


// Crate metadata queries

fn lookup_def(session.session sess, int cnum, vec[ast.ident] path)
        -> option.t[ast.def] {
    auto data = sess.get_external_crate(cnum);

    auto did;
    alt (resolve_path(path, data)) {
        case (rr_ok(?di)) { did = di; }
        case (rr_not_found(?name)) {
            ret none[ast.def];
        }
    }

    auto ebml_r = lookup_item(did._1, data);
    auto kind_ch = get_item_kind(ebml_r);

    did = tup(cnum, did._1);

    // FIXME: It'd be great if we had u8 char literals.
    auto def;
    if (kind_ch == ('c' as u8))         { def = ast.def_const(did);      }
    else if (kind_ch == ('f' as u8))    { def = ast.def_fn(did);         }
    else if (kind_ch == ('F' as u8))    { def = ast.def_native_fn(did);  }
    else if (kind_ch == ('y' as u8))    { def = ast.def_ty(did);         }
    else if (kind_ch == ('o' as u8))    { def = ast.def_obj(did);        }
    else if (kind_ch == ('t' as u8)) {
        // We treat references to tags as references to types.
        def = ast.def_ty(did);
    } else if (kind_ch == ('m' as u8))  { def = ast.def_mod(did);        }
    else if (kind_ch == ('n' as u8))    { def = ast.def_native_mod(did); }
    else if (kind_ch == ('v' as u8)) {
        auto tid = get_variant_tag_id(ebml_r);
        tid = tup(cnum, tid._1);
        def = ast.def_variant(tid, did);
    } else {
        log #fmt("lookup_def(): unknown kind char: %d", kind_ch as int);
        fail;
    }

    ret some[ast.def](def);
}

fn get_type(session.session sess, ast.def_id def) -> ty.ty_params_opt_and_ty {
    auto external_crate_id = def._0;
    auto data = sess.get_external_crate(external_crate_id);
    auto ebml_r = lookup_item(def._1, data);
    auto t = get_item_type(ebml_r, external_crate_id);

    auto tps_opt;
    auto kind_ch = get_item_kind(ebml_r);
    auto has_ty_params = kind_has_type_params(kind_ch);
    if (has_ty_params) {
        auto tps = get_item_ty_params(ebml_r, external_crate_id);
        tps_opt = some[vec[ast.def_id]](tps);
    } else {
        tps_opt = none[vec[ast.def_id]];
    }

    ret tup(tps_opt, t);
}

fn get_symbol(session.session sess, ast.def_id def) -> str {
    auto external_crate_id = def._0;
    auto data = sess.get_external_crate(external_crate_id);
    auto ebml_r = lookup_item(def._1, data);
    ret get_item_symbol(ebml_r);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
