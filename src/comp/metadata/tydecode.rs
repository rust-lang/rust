// Type decoding

import core::{vec, str, uint};
import option::{none, some};
import syntax::ast;
import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::respan;
import middle::ty;

export parse_ty_data, parse_def_id;
export parse_bounds_data;

// Compact string representation for ty::t values. API ty_str &
// parse_from_str. Extra parameters are for converting to/from def_ids in the
// data buffer. Whatever format you choose should not contain pipe characters.

// Callback to translate defs to strs or back:
type conv_did = block(ast::def_id) -> ast::def_id;

type pstate = {data: @[u8], crate: int, mutable pos: uint, tcx: ty::ctxt};

fn peek(st: @pstate) -> u8 {
    st.data[st.pos]
}

fn next(st: @pstate) -> u8 {
    let ch = st.data[st.pos];
    st.pos = st.pos + 1u;
    ret ch;
}

fn parse_ident(st: @pstate, last: char) -> ast::ident {
    fn is_last(b: char, c: char) -> bool { ret c == b; }
    ret parse_ident_(st, bind is_last(last, _));
}

fn parse_ident_(st: @pstate, is_last: fn@(char) -> bool) ->
   ast::ident {
    let rslt = "";
    while !is_last(peek(st) as char) {
        rslt += str::unsafe_from_byte(next(st));
    }
    ret rslt;
}


fn parse_ty_data(data: @[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                 conv: conv_did) -> ty::t {
    let st = @{data: data, crate: crate_num, mutable pos: pos, tcx: tcx};
    parse_ty(st, conv)
}

fn parse_ret_ty(st: @pstate, conv: conv_did) -> (ast::ret_style, ty::t) {
    alt peek(st) as char {
      '!' { next(st); (ast::noreturn, ty::mk_bot(st.tcx)) }
      _ { (ast::return_val, parse_ty(st, conv)) }
    }
}

fn parse_constrs(st: @pstate, conv: conv_did) -> [@ty::constr] {
    let rslt: [@ty::constr] = [];
    alt peek(st) as char {
      ':' {
        do  {
            next(st);
            rslt += [parse_constr(st, conv, parse_constr_arg)];
        } while peek(st) as char == ';'
      }
      _ { }
    }
    ret rslt;
}

// FIXME less copy-and-paste
fn parse_ty_constrs(st: @pstate, conv: conv_did) -> [@ty::type_constr] {
    let rslt: [@ty::type_constr] = [];
    alt peek(st) as char {
      ':' {
        do  {
            next(st);
            rslt += [parse_constr(st, conv, parse_ty_constr_arg)];
        } while peek(st) as char == ';'
      }
      _ { }
    }
    ret rslt;
}

fn parse_path(st: @pstate) -> @ast::path {
    let idents: [ast::ident] = [];
    fn is_last(c: char) -> bool { ret c == '(' || c == ':'; }
    idents += [parse_ident_(st, is_last)];
    while true {
        alt peek(st) as char {
          ':' { next(st); next(st); }
          c {
            if c == '(' {
                ret @respan(ast_util::dummy_sp(),
                            {global: false, idents: idents, types: []});
            } else { idents += [parse_ident_(st, is_last)]; }
          }
        }
    }
    fail "parse_path: ill-formed path";
}

fn parse_constr_arg(st: @pstate) -> ast::fn_constr_arg {
    alt peek(st) as char {
      '*' { st.pos += 1u; ret ast::carg_base; }
      c {

        /* how will we disambiguate between
           an arg index and a lit argument? */
        if c >= '0' && c <= '9' {
            next(st);
            // FIXME
            ret ast::carg_ident((c as uint) - 48u);
        } else {
            #error("Lit args are unimplemented");
            fail; // FIXME
        }
        /*
          else {
          auto lit = parse_lit(st, conv, ',');
          args += [respan(st.span, ast::carg_lit(lit))];
          }
        */
      }
    }
}

fn parse_ty_constr_arg(st: @pstate) -> ast::constr_arg_general_<@path> {
    alt peek(st) as char {
      '*' { st.pos += 1u; ret ast::carg_base; }
      c { ret ast::carg_ident(parse_path(st)); }
    }
}

fn parse_constr<T: copy>(st: @pstate, conv: conv_did,
                         pser: block(@pstate) -> ast::constr_arg_general_<T>)
    -> @ty::constr_general<T> {
    let sp = ast_util::dummy_sp(); // FIXME: use a real span
    let args: [@sp_constr_arg<T>] = [];
    let pth = parse_path(st);
    let ignore: char = next(st) as char;
    assert (ignore == '(');
    let def = parse_def(st, conv);
    let an_arg: constr_arg_general_<T>;
    do  {
        an_arg = pser(st);
        // FIXME use a real span
        args += [@respan(sp, an_arg)];
        ignore = next(st) as char;
    } while ignore == ';'
    assert (ignore == ')');
    ret @respan(sp, {path: pth, args: args, id: def});
}

fn parse_ty_rust_fn(st: @pstate, conv: conv_did, p: ast::proto) -> ty::t {
    ret ty::mk_fn(st.tcx, {proto: p with parse_ty_fn(st, conv)});
}

fn parse_ty(st: @pstate, conv: conv_did) -> ty::t {
    alt next(st) as char {
      'n' { ret ty::mk_nil(st.tcx); }
      'z' { ret ty::mk_bot(st.tcx); }
      'b' { ret ty::mk_bool(st.tcx); }
      'i' { ret ty::mk_int(st.tcx); }
      'u' { ret ty::mk_uint(st.tcx); }
      'l' { ret ty::mk_float(st.tcx); }
      'M' {
        alt next(st) as char {
          'b' { ret ty::mk_mach_uint(st.tcx, ast::ty_u8); }
          'w' { ret ty::mk_mach_uint(st.tcx, ast::ty_u16); }
          'l' { ret ty::mk_mach_uint(st.tcx, ast::ty_u32); }
          'd' { ret ty::mk_mach_uint(st.tcx, ast::ty_u64); }
          'B' { ret ty::mk_mach_int(st.tcx, ast::ty_i8); }
          'W' { ret ty::mk_mach_int(st.tcx, ast::ty_i16); }
          'L' { ret ty::mk_mach_int(st.tcx, ast::ty_i32); }
          'D' { ret ty::mk_mach_int(st.tcx, ast::ty_i64); }
          'f' { ret ty::mk_mach_float(st.tcx, ast::ty_f32); }
          'F' { ret ty::mk_mach_float(st.tcx, ast::ty_f64); }
        }
      }
      'c' { ret ty::mk_char(st.tcx); }
      'S' { ret ty::mk_str(st.tcx); }
      't' {
        assert (next(st) as char == '[');
        let def = parse_def(st, conv);
        let params: [ty::t] = [];
        while peek(st) as char != ']' { params += [parse_ty(st, conv)]; }
        st.pos = st.pos + 1u;
        ret ty::mk_tag(st.tcx, def, params);
      }
      'x' {
        assert (next(st) as char == '[');
        let def = parse_def(st, conv);
        let params: [ty::t] = [];
        while peek(st) as char != ']' { params += [parse_ty(st, conv)]; }
        st.pos = st.pos + 1u;
        ret ty::mk_iface(st.tcx, def, params);
      }
      'p' {
        let did = parse_def(st, conv);
        ret ty::mk_param(st.tcx, parse_int(st) as uint, did);
      }
      '@' { ret ty::mk_box(st.tcx, parse_mt(st, conv)); }
      '~' { ret ty::mk_uniq(st.tcx, parse_mt(st, conv)); }
      '*' { ret ty::mk_ptr(st.tcx, parse_mt(st, conv)); }
      'I' { ret ty::mk_vec(st.tcx, parse_mt(st, conv)); }
      'R' {
        assert (next(st) as char == '[');
        let fields: [ty::field] = [];
        while peek(st) as char != ']' {
            let name = "";
            while peek(st) as char != '=' {
                name += str::unsafe_from_byte(next(st));
            }
            st.pos = st.pos + 1u;
            fields += [{ident: name, mt: parse_mt(st, conv)}];
        }
        st.pos = st.pos + 1u;
        ret ty::mk_rec(st.tcx, fields);
      }
      'T' {
        assert (next(st) as char == '[');
        let params = [];
        while peek(st) as char != ']' { params += [parse_ty(st, conv)]; }
        st.pos = st.pos + 1u;
        ret ty::mk_tup(st.tcx, params);
      }
      's' {
        ret parse_ty_rust_fn(st, conv, ast::proto_send);
      }
      'F' {
        ret parse_ty_rust_fn(st, conv, ast::proto_shared(ast::sugar_normal));
      }
      'f' {
        ret parse_ty_rust_fn(st, conv, ast::proto_bare);
      }
      'B' {
        ret parse_ty_rust_fn(st, conv, ast::proto_block);
      }
      'N' {
        let func = parse_ty_fn(st, conv);
        ret ty::mk_native_fn(st.tcx, func.inputs, func.output);
      }
      'O' {
        assert (next(st) as char == '[');
        let methods: [ty::method] = [];
        while peek(st) as char != ']' {
            let proto;
            alt next(st) as char {
              'f' { proto = ast::proto_bare; }
            }
            let name = "";
            while peek(st) as char != '[' {
                name += str::unsafe_from_byte(next(st));
            }
            methods += [{ident: name, tps: @[],
                         fty: {proto: proto with parse_ty_fn(st, conv)}}];
        }
        st.pos += 1u;
        ret ty::mk_obj(st.tcx, methods);
      }
      'r' {
        assert (next(st) as char == '[');
        let def = parse_def(st, conv);
        let inner = parse_ty(st, conv);
        let params: [ty::t] = [];
        while peek(st) as char != ']' { params += [parse_ty(st, conv)]; }
        st.pos = st.pos + 1u;
        ret ty::mk_res(st.tcx, def, inner, params);
      }
      'X' { ret ty::mk_var(st.tcx, parse_int(st)); }
      'E' { let def = parse_def(st, conv); ret ty::mk_native(st.tcx, def); }
      'Y' { ret ty::mk_type(st.tcx); }
      'y' { ret ty::mk_send_type(st.tcx); }
      'C' { ret ty::mk_opaque_closure(st.tcx); }
      '#' {
        let pos = parse_hex(st);
        assert (next(st) as char == ':');
        let len = parse_hex(st);
        assert (next(st) as char == '#');
        alt st.tcx.rcache.find({cnum: st.crate, pos: pos, len: len}) {
          some(tt) { ret tt; }
          none. {
            let ps = @{pos: pos with *st};
            let tt = parse_ty(ps, conv);
            st.tcx.rcache.insert({cnum: st.crate, pos: pos, len: len}, tt);
            ret tt;
          }
        }
      }
      'A' {
        assert (next(st) as char == '[');
        let tt = parse_ty(st, conv);
        let tcs = parse_ty_constrs(st, conv);
        assert (next(st) as char == ']');
        ret ty::mk_constr(st.tcx, tt, tcs);
      }
      c { #error("unexpected char in type string: %c", c); fail;}
    }
}

fn parse_mt(st: @pstate, conv: conv_did) -> ty::mt {
    let m;
    alt peek(st) as char {
      'm' { next(st); m = ast::mut; }
      '?' { next(st); m = ast::maybe_mut; }
      _ { m = ast::imm; }
    }
    ret {ty: parse_ty(st, conv), mut: m};
}

fn parse_def(st: @pstate, conv: conv_did) -> ast::def_id {
    let def = [];
    while peek(st) as char != '|' { def += [next(st)]; }
    st.pos = st.pos + 1u;
    ret conv(parse_def_id(def));
}

fn parse_int(st: @pstate) -> int {
    let n = 0;
    while true {
        let cur = peek(st) as char;
        if cur < '0' || cur > '9' { break; }
        st.pos = st.pos + 1u;
        n *= 10;
        n += (cur as int) - ('0' as int);
    }
    ret n;
}

fn parse_hex(st: @pstate) -> uint {
    let n = 0u;
    while true {
        let cur = peek(st) as char;
        if (cur < '0' || cur > '9') && (cur < 'a' || cur > 'f') { break; }
        st.pos = st.pos + 1u;
        n *= 16u;
        if '0' <= cur && cur <= '9' {
            n += (cur as uint) - ('0' as uint);
        } else { n += 10u + (cur as uint) - ('a' as uint); }
    }
    ret n;
}

fn parse_ty_fn(st: @pstate, conv: conv_did) -> ty::fn_ty {
    assert (next(st) as char == '[');
    let inputs: [ty::arg] = [];
    while peek(st) as char != ']' {
        let mode = alt peek(st) as char {
          '&' { ast::by_mut_ref }
          '-' { ast::by_move }
          '+' { ast::by_copy }
          '=' { ast::by_ref }
          '#' { ast::by_val }
        };
        st.pos += 1u;
        inputs += [{mode: mode, ty: parse_ty(st, conv)}];
    }
    st.pos += 1u; // eat the ']'
    let cs = parse_constrs(st, conv);
    let (ret_style, ret_ty) = parse_ret_ty(st, conv);
    ret {proto: ast::proto_bare, inputs: inputs, output: ret_ty,
         ret_style: ret_style, constraints: cs};
}


// Rust metadata parsing
fn parse_def_id(buf: [u8]) -> ast::def_id {
    let colon_idx = 0u;
    let len = vec::len(buf);
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1u; }
    if colon_idx == len {
        #error("didn't find ':' when parsing def id");
        fail;
    }
    let crate_part = vec::slice::<u8>(buf, 0u, colon_idx);
    let def_part = vec::slice::<u8>(buf, colon_idx + 1u, len);

    let crate_part_vec = [];
    let def_part_vec = [];
    for b: u8 in crate_part { crate_part_vec += [b]; }
    for b: u8 in def_part { def_part_vec += [b]; }

    let crate_num = uint::parse_buf(crate_part_vec, 10u) as int;
    let def_num = uint::parse_buf(def_part_vec, 10u) as int;
    ret {crate: crate_num, node: def_num};
}

fn parse_bounds_data(data: @[u8], start: uint,
                     crate_num: int, tcx: ty::ctxt, conv: conv_did)
    -> @[ty::param_bound] {
    let st = @{data: data, crate: crate_num, mutable pos: start, tcx: tcx};
    parse_bounds(st, conv)
}

fn parse_bounds(st: @pstate, conv: conv_did) -> @[ty::param_bound] {
    let bounds = [];
    while true {
        bounds += [alt next(st) as char {
          'S' { ty::bound_send }
          'C' { ty::bound_copy }
          'I' { ty::bound_iface(parse_ty(st, conv)) }
          '.' { break; }
        }];
    }
    @bounds
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
