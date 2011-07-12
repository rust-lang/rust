// Type decoding

import std::ivec;
import std::str;
import std::vec;
import std::uint;
import std::option;
import std::option::none;
import std::option::some;
import syntax::ast;
import ast::respan;
import middle::ty;

export parse_def_id;
export parse_ty_data;

// Compact string representation for ty::t values. API ty_str &
// parse_from_str. Extra parameters are for converting to/from def_ids in the
// data buffer. Whatever format you choose should not contain pipe characters.

// Callback to translate defs to strs or back:
type str_def = fn(str) -> ast::def_id ;

type pstate =
    rec(@u8[] data, int crate, mutable uint pos, uint len, ty::ctxt tcx);

tag ty_or_bang { a_ty(ty::t); a_bang; }

fn peek(@pstate st) -> u8 { ret st.data.(st.pos); }

fn next(@pstate st) -> u8 {
    auto ch = st.data.(st.pos);
    st.pos = st.pos + 1u;
    ret ch;
}

fn parse_ident(@pstate st, str_def sd, char last) -> ast::ident {
    fn is_last(char b, char c) -> bool {
        ret c == b;
    }
    ret parse_ident_(st, sd, bind is_last(last, _));
}

fn parse_ident_(@pstate st, str_def sd, fn(char) -> bool is_last)
    -> ast::ident {
    auto rslt = "";
    while (! is_last(peek(st) as char)) {
        rslt += str::unsafe_from_byte(next(st));
    }
    ret rslt;
}


fn parse_ty_data(@u8[] data, int crate_num, uint pos, uint len, str_def sd,
                 ty::ctxt tcx) -> ty::t {
    auto st =
        @rec(data=data, crate=crate_num, mutable pos=pos, len=len, tcx=tcx);
    auto result = parse_ty(st, sd);
    ret result;
}

fn parse_ty_or_bang(@pstate st, str_def sd) -> ty_or_bang {
    alt (peek(st) as char) {
        case ('!') { next(st); ret a_bang; }
        case (_) { ret a_ty(parse_ty(st, sd)); }
    }
}

fn parse_constrs(@pstate st, str_def sd) -> (@ty::constr_def)[] {
    let (@ty::constr_def)[] rslt = ~[];
    alt (peek(st) as char) {
        case (':') {
            do  {
                next(st);
                rslt += ~[parse_constr(st, sd)];
            } while (peek(st) as char == ';')
        }
        case (_) { }
    }
    ret rslt;
}

fn parse_path(@pstate st, str_def sd) -> ast::path {
    let ast::ident[] idents = ~[];
    fn is_last(char c) -> bool {
        ret (c == '(' || c == ':');
    }
    idents += ~[parse_ident_(st, sd, is_last)];
    while (true) {
        alt (peek(st) as char) {
            case (':') { next(st); next(st); }
            case (?c) {
                if (c == '(') {
                    ret respan(rec(lo=0u, hi=0u),
                               rec(global=false, idents=idents, types=~[]));
                }
                else {
                    idents += ~[parse_ident_(st, sd, is_last)];
                }
            }
        }
    }
    fail "parse_path: ill-formed path";
}

fn parse_constr(@pstate st, str_def sd) -> @ty::constr_def {
    let (@ast::constr_arg)[] args = ~[];
    auto sp = rec(lo=0u,hi=0u); // FIXME: use a real span
    let ast::path pth = parse_path(st, sd);
    let char ignore = next(st) as char;
    assert(ignore as char == '(');
    auto def = parse_def(st, sd);
    do {
        alt (peek(st) as char) {
            case ('*') {
                st.pos += 1u;
                args += ~[@respan(sp, ast::carg_base)];
            }
            case (?c) {
                /* how will we disambiguate between
                 an arg index and a lit argument? */
                if (c >= '0' && c <= '9') {
                    // FIXME
                    args += ~[@respan(sp,
                                      ast::carg_ident((c as uint) - 48u))];
                    ignore = next(st) as char;
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
        ignore = next(st) as char;
    } while (ignore == ';');
    assert(ignore == ')');
    ret @respan(sp, rec(path=pth, args=args, id=def));
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
                case ('b') { ret ty::mk_mach(st.tcx, ast::ty_u8); }
                case ('w') { ret ty::mk_mach(st.tcx, ast::ty_u16); }
                case ('l') { ret ty::mk_mach(st.tcx, ast::ty_u32); }
                case ('d') { ret ty::mk_mach(st.tcx, ast::ty_u64); }
                case ('B') { ret ty::mk_mach(st.tcx, ast::ty_i8); }
                case ('W') { ret ty::mk_mach(st.tcx, ast::ty_i16); }
                case ('L') { ret ty::mk_mach(st.tcx, ast::ty_i32); }
                case ('D') { ret ty::mk_mach(st.tcx, ast::ty_i64); }
                case ('f') { ret ty::mk_mach(st.tcx, ast::ty_f32); }
                case ('F') { ret ty::mk_mach(st.tcx, ast::ty_f64); }
            }
        }
        case ('c') { ret ty::mk_char(st.tcx); }
        case ('s') { ret ty::mk_str(st.tcx); }
        case ('S') { ret ty::mk_istr(st.tcx); }
        case ('t') {
            assert (next(st) as char == '[');
            auto def = parse_def(st, sd);
            let ty::t[] params = ~[];
            while (peek(st) as char != ']') { params += ~[parse_ty(st, sd)]; }
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
            let ty::mt[] params = ~[];
            while (peek(st) as char != ']') { params += ~[parse_mt(st, sd)]; }
            st.pos = st.pos + 1u;
            ret ty::mk_tup(st.tcx, params);
        }
        case ('R') {
            assert (next(st) as char == '[');
            let ty::field[] fields = ~[];
            while (peek(st) as char != ']') {
                auto name = "";
                while (peek(st) as char != '=') {
                    name += str::unsafe_from_byte(next(st));
                }
                st.pos = st.pos + 1u;
                fields += ~[rec(ident=name, mt=parse_mt(st, sd))];
            }
            st.pos = st.pos + 1u;
            ret ty::mk_rec(st.tcx, fields);
        }
        case ('F') {
            auto func = parse_ty_fn(st, sd);
            ret ty::mk_fn(st.tcx, ast::proto_fn, func._0, func._1, func._2,
                          func._3);
        }
        case ('W') {
            auto func = parse_ty_fn(st, sd);
            ret ty::mk_fn(st.tcx, ast::proto_iter, func._0, func._1, func._2,
                          func._3);
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
            ret ty::mk_native_fn(st.tcx, abi, func._0, func._1);
        }
        case ('O') {
            assert (next(st) as char == '[');
            let ty::method[] methods = ~[];
            while (peek(st) as char != ']') {
                auto proto;
                alt (next(st) as char) {
                    case ('W') { proto = ast::proto_iter; }
                    case ('F') { proto = ast::proto_fn; }
                }
                auto name = "";
                while (peek(st) as char != '[') {
                    name += str::unsafe_from_byte(next(st));
                }
                auto func = parse_ty_fn(st, sd);
                methods +=
                    ~[rec(proto=proto,
                          ident=name,
                          inputs=func._0,
                          output=func._1,
                          cf=func._2,
                          constrs=func._3)];
            }
            st.pos += 1u;
            ret ty::mk_obj(st.tcx, methods);
        }
        case ('r') {
            assert (next(st) as char == '[');
            auto def = parse_def(st, sd);
            auto inner = parse_ty(st, sd);
            let ty::t[] params = ~[];
            while (peek(st) as char != ']') { params += ~[parse_ty(st, sd)]; }
            st.pos = st.pos + 1u;
            ret ty::mk_res(st.tcx, def, inner, params);
        }
        case ('X') { ret ty::mk_var(st.tcx, parse_int(st)); }
        case ('E') {
            auto def = parse_def(st, sd);
            ret ty::mk_native(st.tcx, def);
        }
        case ('Y') { ret ty::mk_type(st.tcx); }
        case ('#') {
            auto pos = parse_hex(st);
            assert (next(st) as char == ':');
            auto len = parse_hex(st);
            assert (next(st) as char == '#');
            alt (st.tcx.rcache.find(tup(st.crate, pos, len))) {
                case (some(?tt)) { ret tt; }
                case (none) {
                    auto ps = @rec(pos=pos, len=len with *st);
                    auto tt = parse_ty(ps, sd);
                    st.tcx.rcache.insert(tup(st.crate, pos, len), tt);
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
        case ('m') { next(st); mut = ast::mut; }
        case ('?') { next(st); mut = ast::maybe_mut; }
        case (_) { mut = ast::imm; }
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
        if (cur < '0' || cur > '9') { break; }
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
        if ((cur < '0' || cur > '9') && (cur < 'a' || cur > 'f')) { break; }
        st.pos = st.pos + 1u;
        n *= 16u;
        if ('0' <= cur && cur <= '9') {
            n += (cur as uint) - ('0' as uint);
        } else { n += 10u + (cur as uint) - ('a' as uint); }
    }
    ret n;
}

fn parse_ty_fn(@pstate st, str_def sd) ->
   tup(ty::arg[], ty::t, ast::controlflow, (@ty::constr_def)[]) {
    assert (next(st) as char == '[');
    let ty::arg[] inputs = ~[];
    while (peek(st) as char != ']') {
        auto mode = ty::mo_val;
        if (peek(st) as char == '&') {
            mode = ty::mo_alias(false);
            st.pos += 1u;
            if (peek(st) as char == 'm') {
                mode = ty::mo_alias(true);
                st.pos += 1u;
            }
        }
        inputs += ~[rec(mode=mode, ty=parse_ty(st, sd))];
    }
    st.pos += 1u; // eat the ']'
    auto cs = parse_constrs(st, sd);
    alt (parse_ty_or_bang(st, sd)) {
        case (a_bang) {
            ret tup(inputs, ty::mk_bot(st.tcx), ast::noreturn, cs);
        }
        case (a_ty(?t)) { ret tup(inputs, t, ast::return, cs); }
    }
}


// Rust metadata parsing
fn parse_def_id(&u8[] buf) -> ast::def_id {
    auto colon_idx = 0u;
    auto len = ivec::len[u8](buf);
    while (colon_idx < len && buf.(colon_idx) != ':' as u8) {
        colon_idx += 1u;
    }
    if (colon_idx == len) {
        log_err "didn't find ':' when parsing def id";
        fail;
    }
    auto crate_part = ivec::slice[u8](buf, 0u, colon_idx);
    auto def_part = ivec::slice[u8](buf, colon_idx + 1u, len);

    // FIXME: Remove these ivec->vec conversions.
    auto crate_part_vec = []; auto def_part_vec = [];
    for (u8 b in crate_part) { crate_part_vec += [b]; }
    for (u8 b in def_part) { def_part_vec += [b]; }

    auto crate_num = uint::parse_buf(crate_part_vec, 10u) as int;
    auto def_id = uint::parse_buf(def_part_vec, 10u) as int;
    ret tup(crate_num, def_id);
}
