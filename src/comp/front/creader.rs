// -*- rust -*-

import driver.session;
import front.ast;
import lib.llvm.llvmext;
import lib.llvm.mk_object_file;
import lib.llvm.mk_section_iter;
import middle.fold;
import middle.ty;
import util.common;
import util.common.span;

import std._str;
import std._vec;
import std.fs;
import std.option;
import std.os;
import std.map.hashmap;

// TODO: map to a real type here.
type env = @rec(
    @hashmap[str, @ast.external_crate_info] crate_cache,
    vec[str] library_search_paths
);

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



// TODO: return something
fn load_crate(ast.ident ident, vec[str] library_search_paths) -> @() {
    for (str library_search_path in library_search_paths) {
        auto path = fs.connect(library_search_path, ident);
        // TODO
    }

    ret @();
}

fn fold_view_item_use(&env e, &span sp, ast.ident ident,
        vec[@ast.meta_item] meta_items, ast.def_id id, ast.ann orig_ann)
        -> @ast.view_item {
    auto external_crate;
    if (!e.crate_cache.contains_key(ident)) {
        external_crate = load_crate(ident, e.library_search_paths);
        e.crate_cache.insert(ident, external_crate);
    } else {
        external_crate = e.crate_cache.get(ident);
    }

    auto ann = ast.ann_crate(external_crate);
    auto viu = ast.view_item_use(ident, meta_items, id, ann);
    ret @fold.respan[ast.view_item_](sp, viu);
}

// Reads external crates referenced by "use" directives.
fn read_crates(session.session sess,
               @ast.crate crate,
               vec[str] library_search_paths) -> @ast.crate {
    auto e = @rec(
        crate_cache=@common.new_str_hash[@ast.external_crate_info](),
        library_search_paths=library_search_paths
    );

    auto f = fold_view_item_use;
    auto fld = @rec(fold_view_item_use=f with *fold.new_identity_fold[env]());
    ret fold.fold_crate[env](e, fld, crate);
}

