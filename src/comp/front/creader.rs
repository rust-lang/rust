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
import middle.typeck;
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
    rr_not_found(vec[ast.ident], ast.ident);
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
            auto def = "";
            while (peek(st) as char != '|') {
                def += _str.unsafe_from_byte(next(st));
            }
            st.pos = st.pos + 1u;
            let vec[@ty.t] params = vec();
            while (peek(st) as char != ']') {
                params += vec(parse_ty(st, sd));
            }
            st.pos = st.pos + 1u;
            ret ty.ty_tag(sd(def), params);
        }
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

// Given a path and serialized crate metadata, returns the ID of the
// definition the path refers to.
impure fn resolve_path(vec[ast.ident] path, vec[u8] data) -> resolve_result {
    impure fn resolve_path_inner(vec[ast.ident] path, &ebml.reader ebml_r)
            -> resolve_result {
        auto i = 0u;
        auto len = _vec.len[ast.ident](path);
        while (i < len) {
            auto name = path.(i);
            auto last = i == len - 1u;

            // Search this level for the identifier.
            auto found = false;
            while (ebml.bytes_left(ebml_r) > 0u && !found) {
                auto ebml_tag = ebml.peek(ebml_r);
                check ((ebml_tag.id == metadata.tag_paths_item) ||
                       (ebml_tag.id == metadata.tag_paths_mod));

                ebml.move_to_first_child(ebml_r);
                auto did_opt = none[ast.def_id];
                auto name_opt = none[ast.ident];
                while (ebml.bytes_left(ebml_r) > 0u) {
                    auto inner_tag = ebml.peek(ebml_r);
                    if (inner_tag.id == metadata.tag_paths_name) {
                        ebml.move_to_first_child(ebml_r);
                        auto name_data = ebml.read_data(ebml_r);
                        ebml.move_to_parent(ebml_r);
                        auto nm = _str.unsafe_from_bytes(name_data);
                        name_opt = some[ast.ident](nm);
                    } else if (inner_tag.id == metadata.tag_items_def_id) {
                        ebml.move_to_first_child(ebml_r);
                        auto did_data = ebml.read_data(ebml_r);
                        ebml.move_to_parent(ebml_r);
                        did_opt = some[ast.def_id](parse_def_id(did_data));
                    }
                    ebml.move_to_next_sibling(ebml_r);
                }
                ebml.move_to_parent(ebml_r);

                if (_str.eq(option.get[ast.ident](name_opt), name)) {
                    // Matched!
                    if (last) {
                        ret rr_ok(option.get[ast.def_id](did_opt));
                    }

                    // Move to the module/item we found for the next iteration
                    // of the loop...
                    ebml.move_to_first_child(ebml_r);
                    found = true;
                }

                ebml.move_to_next_sibling(ebml_r);
            }

            if (!found) {
                auto prev = _vec.slice[ast.ident](path, 0u, i);
                ret rr_not_found(prev, name);
            }

            i += 1u;
        }

        fail;   // not reached
    }

    auto io_r = io.new_reader_(io.new_byte_buf_reader(data));
    auto ebml_r = ebml.create_reader(io_r);
    while (ebml.bytes_left(ebml_r) > 0u) {
        auto ebml_tag = ebml.peek(ebml_r);
        if (ebml_tag.id == metadata.tag_paths) {
            ebml.move_to_first_child(ebml_r);
            ret resolve_path_inner(path, ebml_r);
        }
        ebml.move_to_next_sibling(ebml_r);
    }

    log "resolve_path(): no names in file";
    fail;
}

impure fn move_to_item(&ebml.reader ebml_r, ast.def_id did) {
    while (ebml.bytes_left(ebml_r) > 0u) {
        auto outer_ebml_tag = ebml.peek(ebml_r);
        if (outer_ebml_tag.id == metadata.tag_items) {
            ebml.move_to_first_child(ebml_r);

            while (ebml.bytes_left(ebml_r) > 0u) {
                auto inner_ebml_tag = ebml.peek(ebml_r);
                if (inner_ebml_tag.id == metadata.tag_items_item) {
                    ebml.move_to_first_child(ebml_r);

                    while (ebml.bytes_left(ebml_r) > 0u) {
                        auto innermost_ebml_tag = ebml.peek(ebml_r);
                        if (innermost_ebml_tag.id ==
                                metadata.tag_items_def_id) {
                            ebml.move_to_first_child(ebml_r);
                            auto did_data = ebml.read_data(ebml_r);
                            ebml.move_to_parent(ebml_r);

                            auto this_did = parse_def_id(did_data);
                            if (did._0 == this_did._0 &&
                                    did._1 == this_did._1) {
                                // Move to the start of this item's data.
                                ebml.move_to_parent(ebml_r);
                                ebml.move_to_first_child(ebml_r);
                                ret;
                            }
                        }
                        ebml.move_to_next_sibling(ebml_r);
                    }
                    ebml.move_to_parent(ebml_r);
                }
                ebml.move_to_next_sibling(ebml_r);
            }
            ebml.move_to_parent(ebml_r);
        }
        ebml.move_to_next_sibling(ebml_r);
    }

    log #fmt("move_to_item: item not found: %d:%d", did._0, did._1);
}

// Looks up an item in the given metadata and returns an EBML reader pointing
// to the item data.
impure fn lookup_item(ast.def_id did, vec[u8] data) -> ebml.reader {
    auto io_r = io.new_reader_(io.new_byte_buf_reader(data));
    auto ebml_r = ebml.create_reader(io_r);
    move_to_item(ebml_r, did);
    ret ebml_r;
}

impure fn get_item_kind(&ebml.reader ebml_r) -> u8 {
    while (ebml.bytes_left(ebml_r) > 0u) {
        auto ebml_tag = ebml.peek(ebml_r);
        if (ebml_tag.id == metadata.tag_items_kind) {
            ebml.move_to_first_child(ebml_r);
            auto kind_ch = ebml.read_data(ebml_r).(0);

            // Reset the EBML reader so the callee can use it to look up
            // additional info about the item.
            ebml.move_to_parent(ebml_r);
            ebml.move_to_parent(ebml_r);
            ebml.move_to_first_child(ebml_r);

            ret kind_ch;
        }
        ebml.move_to_next_sibling(ebml_r);
    }

    log "get_item_kind(): no kind found";
    fail;
}

impure fn get_variant_tag_id(&ebml.reader ebml_r) -> ast.def_id {
    while (ebml.bytes_left(ebml_r) > 0u) {
        auto ebml_tag = ebml.peek(ebml_r);
        if (ebml_tag.id == metadata.tag_items_tag_id) {
            ebml.move_to_first_child(ebml_r);
            auto tid = parse_def_id(ebml.read_data(ebml_r));

            // Be kind, rewind.
            ebml.move_to_parent(ebml_r);
            ebml.move_to_parent(ebml_r);
            ebml.move_to_first_child(ebml_r);

            ret tid;
        }
    }

    log "get_variant_tag_id(): no tag ID found";
    fail;
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


// Crate metadata queries

fn lookup_def(session.session sess, &span sp, int cnum, vec[ast.ident] path)
        -> ast.def {
    auto data = sess.get_external_crate(cnum);

    auto did;
    alt (resolve_path(path, data)) {
        case (rr_ok(?di)) { did = di; }
        case (rr_not_found(?prev, ?name)) {
            sess.span_err(sp,
                #fmt("unbound name '%s' (no item named '%s' found in '%s')",
                     _str.connect(path, "."), name, _str.connect(prev, ".")));
            fail;
        }
    }

    auto ebml_r = lookup_item(did, data);
    auto kind_ch = get_item_kind(ebml_r);

    did = tup(cnum, did._1);

    // FIXME: It'd be great if we had u8 char literals.
    if (kind_ch == ('c' as u8))      { ret ast.def_const(did);  }
    else if (kind_ch == ('f' as u8)) { ret ast.def_fn(did);     }
    else if (kind_ch == ('y' as u8)) { ret ast.def_ty(did);     }
    else if (kind_ch == ('o' as u8)) { ret ast.def_obj(did);    }
    else if (kind_ch == ('t' as u8)) { ret ast.def_ty(did);     }
    else if (kind_ch == ('v' as u8)) {
        auto tid = get_variant_tag_id(ebml_r);
        tid = tup(cnum, tid._1);
        ret ast.def_variant(tid, did);
    }

    log #fmt("lookup_def(): unknown kind char: %d", kind_ch as int);
    fail;
}

fn get_type(session.session sess, ast.def_id def) -> typeck.ty_and_params {
    // FIXME: fill in.
    fail;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
