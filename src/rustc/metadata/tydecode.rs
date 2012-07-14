// Type decoding

import syntax::ast;
import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::respan;
import middle::ty;
import std::map::hashmap;

export parse_ty_data, parse_def_id, parse_ident;
export parse_bounds_data;

// Compact string representation for ty::t values. API ty_str &
// parse_from_str. Extra parameters are for converting to/from def_ids in the
// data buffer. Whatever format you choose should not contain pipe characters.

// Callback to translate defs to strs or back:
type conv_did = fn(ast::def_id) -> ast::def_id;

type pstate = {data: @~[u8], crate: int, mut pos: uint, tcx: ty::ctxt};

fn peek(st: @pstate) -> char {
    st.data[st.pos] as char
}

fn next(st: @pstate) -> char {
    let ch = st.data[st.pos] as char;
    st.pos = st.pos + 1u;
    ret ch;
}

fn next_byte(st: @pstate) -> u8 {
    let b = st.data[st.pos];
    st.pos = st.pos + 1u;
    ret b;
}

fn parse_ident(st: @pstate, last: char) -> ast::ident {
    fn is_last(b: char, c: char) -> bool { ret c == b; }
    ret parse_ident_(st, |a| is_last(last, a) );
}

fn parse_ident_(st: @pstate, is_last: fn@(char) -> bool) ->
   ast::ident {
    let mut rslt = ~"";
    while !is_last(peek(st)) {
        rslt += str::from_byte(next_byte(st));
    }
    ret @rslt;
}


fn parse_ty_data(data: @~[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                 conv: conv_did) -> ty::t {
    let st = @{data: data, crate: crate_num, mut pos: pos, tcx: tcx};
    parse_ty(st, conv)
}

fn parse_ret_ty(st: @pstate, conv: conv_did) -> (ast::ret_style, ty::t) {
    alt peek(st) {
      '!' { next(st); (ast::noreturn, ty::mk_bot(st.tcx)) }
      _ { (ast::return_val, parse_ty(st, conv)) }
    }
}

fn parse_constrs_gen<T: copy>(st: @pstate, conv: conv_did,
                                       pser: fn(@pstate)
  -> ast::constr_arg_general_<T>) -> ~[@ty::constr_general<T>] {
    let mut rslt: ~[@ty::constr_general<T>] = ~[];
    alt peek(st) {
      ':' {
        loop {
          next(st);
          vec::push(rslt, parse_constr(st, conv, pser));
          if peek(st) != ';' { break; }
        }
      }
      _ {}
    }
    rslt
}

fn parse_constrs(st: @pstate, conv: conv_did) -> ~[@ty::constr] {
    parse_constrs_gen(st, conv, parse_constr_arg)
}

fn parse_ty_constrs(st: @pstate, conv: conv_did) -> ~[@ty::type_constr] {
    parse_constrs_gen(st, conv, parse_ty_constr_arg)
}

fn parse_path(st: @pstate) -> @ast::path {
    let mut idents: ~[ast::ident] = ~[];
    fn is_last(c: char) -> bool { ret c == '(' || c == ':'; }
    vec::push(idents, parse_ident_(st, is_last));
    loop {
        alt peek(st) {
          ':' { next(st); next(st); }
          c {
            if c == '(' {
                ret @{span: ast_util::dummy_sp(),
                      global: false, idents: idents,
                      rp: none, types: ~[]};
            } else { vec::push(idents, parse_ident_(st, is_last)); }
          }
        }
    };
}

fn parse_constr_arg(st: @pstate) -> ast::fn_constr_arg {
    alt peek(st) {
      '*' { st.pos += 1u; ret ast::carg_base; }
      c {

        /* how will we disambiguate between
           an arg index and a lit argument? */
        if c >= '0' && c <= '9' {
            next(st);
            // FIXME #877
            ret ast::carg_ident((c as uint) - 48u);
        } else {
            #error("Lit args are unimplemented");
            fail; // FIXME #877
        }
        /*
          else {
          auto lit = parse_lit(st, conv, ',');
          vec::push(args, respan(st.span, ast::carg_lit(lit)));
          }
        */
      }
    }
}

fn parse_ty_constr_arg(st: @pstate) -> ast::constr_arg_general_<@path> {
    alt peek(st) {
      '*' { st.pos += 1u; ret ast::carg_base; }
      c { ret ast::carg_ident(parse_path(st)); }
    }
}

fn parse_constr<T: copy>(st: @pstate, conv: conv_did,
                         pser: fn(@pstate) -> ast::constr_arg_general_<T>)
    -> @ty::constr_general<T> {
    // FIXME: use real spans and not a bogus one (#2407)
    let sp = ast_util::dummy_sp();
    let mut args: ~[@sp_constr_arg<T>] = ~[];
    let pth = parse_path(st);
    let mut ignore: char = next(st);
    assert (ignore == '(');
    let def = parse_def(st, conv);
    let mut an_arg: constr_arg_general_<T>;
    loop {
        an_arg = pser(st);
        vec::push(args, @respan(sp, an_arg));
        ignore = next(st);
        if ignore != ';' { break; }
    }
    assert (ignore == ')');
    ret @respan(sp, {path: pth, args: args, id: def});
}

fn parse_ty_rust_fn(st: @pstate, conv: conv_did) -> ty::t {
    ret ty::mk_fn(st.tcx, parse_ty_fn(st, conv));
}

fn parse_proto(c: char) -> ast::proto {
    alt c {
      '~' { ast::proto_uniq }
      '@' { ast::proto_box }
      '*' { ast::proto_any }
      '&' { ast::proto_block }
      'n' { ast::proto_bare }
      _ { fail ~"illegal fn type kind " + str::from_char(c); }
    }
}

fn parse_vstore(st: @pstate) -> ty::vstore {
    assert next(st) == '/';

    let c = peek(st);
    if '0' <= c && c <= '9' {
        let n = parse_int(st) as uint;
        assert next(st) == '|';
        ret ty::vstore_fixed(n);
    }

    alt check next(st) {
      '~' { ty::vstore_uniq }
      '@' { ty::vstore_box }
      '&' { ty::vstore_slice(parse_region(st)) }
    }
}

fn parse_substs(st: @pstate, conv: conv_did) -> ty::substs {
    let self_r = parse_opt(st, || parse_region(st) );

    let self_ty = parse_opt(st, || parse_ty(st, conv) );

    assert next(st) == '[';
    let mut params: ~[ty::t] = ~[];
    while peek(st) != ']' { vec::push(params, parse_ty(st, conv)); }
    st.pos = st.pos + 1u;

    ret {self_r: self_r,
         self_ty: self_ty,
         tps: params};
}

fn parse_bound_region(st: @pstate) -> ty::bound_region {
    alt check next(st) {
      's' { ty::br_self }
      'a' { ty::br_anon }
      '[' { ty::br_named(@parse_str(st, ']')) }
    }
}

fn parse_region(st: @pstate) -> ty::region {
    alt check next(st) {
      'b' {
        ty::re_bound(parse_bound_region(st))
      }
      'f' {
        assert next(st) == '[';
        let id = parse_int(st);
        assert next(st) == '|';
        let br = parse_bound_region(st);
        assert next(st) == ']';
        ty::re_free(id, br)
      }
      's' {
        let id = parse_int(st);
        assert next(st) == '|';
        ty::re_scope(id)
      }
      't' {
        ty::re_static
      }
    }
}

fn parse_opt<T>(st: @pstate, f: fn() -> T) -> option<T> {
    alt check next(st) {
      'n' { none }
      's' { some(f()) }
    }
}

fn parse_str(st: @pstate, term: char) -> ~str {
    let mut result = ~"";
    while peek(st) != term {
        result += str::from_byte(next_byte(st));
    }
    next(st);
    ret result;
}

fn parse_ty(st: @pstate, conv: conv_did) -> ty::t {
    alt check next(st) {
      'n' { ret ty::mk_nil(st.tcx); }
      'z' { ret ty::mk_bot(st.tcx); }
      'b' { ret ty::mk_bool(st.tcx); }
      'i' { ret ty::mk_int(st.tcx); }
      'u' { ret ty::mk_uint(st.tcx); }
      'l' { ret ty::mk_float(st.tcx); }
      'M' {
        alt check next(st) {
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
        assert (next(st) == '[');
        let def = parse_def(st, conv);
        let substs = parse_substs(st, conv);
        assert next(st) == ']';
        ret ty::mk_enum(st.tcx, def, substs);
      }
      'x' {
        assert next(st) == '[';
        let def = parse_def(st, conv);
        let substs = parse_substs(st, conv);
        assert next(st) == ']';
        ret ty::mk_trait(st.tcx, def, substs);
      }
      'p' {
        let did = parse_def(st, conv);
        ret ty::mk_param(st.tcx, parse_int(st) as uint, did);
      }
      's' {
        ret ty::mk_self(st.tcx);
      }
      '@' { ret ty::mk_box(st.tcx, parse_mt(st, conv)); }
      '~' { ret ty::mk_uniq(st.tcx, parse_mt(st, conv)); }
      '*' { ret ty::mk_ptr(st.tcx, parse_mt(st, conv)); }
      '&' {
        let r = parse_region(st);
        let mt = parse_mt(st, conv);
        ret ty::mk_rptr(st.tcx, r, mt);
      }
      'I' { ret ty::mk_vec(st.tcx, parse_mt(st, conv)); }
      'U' { ret ty::mk_unboxed_vec(st.tcx, parse_mt(st, conv)); }
      'V' {
        let mt = parse_mt(st, conv);
        let v = parse_vstore(st);
        ret ty::mk_evec(st.tcx, mt, v);
      }
      'v' {
        let v = parse_vstore(st);
        ret ty::mk_estr(st.tcx, v);
      }
      'R' {
        assert (next(st) == '[');
        let mut fields: ~[ty::field] = ~[];
        while peek(st) != ']' {
            let name = @parse_str(st, '=');
            vec::push(fields, {ident: name, mt: parse_mt(st, conv)});
        }
        st.pos = st.pos + 1u;
        ret ty::mk_rec(st.tcx, fields);
      }
      'T' {
        assert (next(st) == '[');
        let mut params = ~[];
        while peek(st) != ']' { vec::push(params, parse_ty(st, conv)); }
        st.pos = st.pos + 1u;
        ret ty::mk_tup(st.tcx, params);
      }
      'f' {
        parse_ty_rust_fn(st, conv)
      }
      'X' {
        ret ty::mk_var(st.tcx, ty::tv_vid(parse_int(st) as uint));
      }
      'Y' { ret ty::mk_type(st.tcx); }
      'C' {
        let ck = alt check next(st) {
          '&' { ty::ck_block }
          '@' { ty::ck_box }
          '~' { ty::ck_uniq }
        };
        ret ty::mk_opaque_closure_ptr(st.tcx, ck);
      }
      '#' {
        let pos = parse_hex(st);
        assert (next(st) == ':');
        let len = parse_hex(st);
        assert (next(st) == '#');
        alt st.tcx.rcache.find({cnum: st.crate, pos: pos, len: len}) {
          some(tt) { ret tt; }
          none {
            let ps = @{pos: pos with *st};
            let tt = parse_ty(ps, conv);
            st.tcx.rcache.insert({cnum: st.crate, pos: pos, len: len}, tt);
            ret tt;
          }
        }
      }
      'A' {
        assert (next(st) == '[');
        let tt = parse_ty(st, conv);
        let tcs = parse_ty_constrs(st, conv);
        assert (next(st) == ']');
        ret ty::mk_constr(st.tcx, tt, tcs);
      }
      '"' {
        let def = parse_def(st, conv);
        let inner = parse_ty(st, conv);
        ty::mk_with_id(st.tcx, inner, def)
      }
      'B' { ty::mk_opaque_box(st.tcx) }
      'a' {
          #debug("saw a class");
          assert (next(st) == '[');
          #debug("saw a [");
          let did = parse_def(st, conv);
          #debug("parsed a def_id %?", did);
          let substs = parse_substs(st, conv);
          assert (next(st) == ']');
          ret ty::mk_class(st.tcx, did, substs);
      }
      c { #error("unexpected char in type string: %c", c); fail;}
    }
}

fn parse_mt(st: @pstate, conv: conv_did) -> ty::mt {
    let mut m;
    alt peek(st) {
      'm' { next(st); m = ast::m_mutbl; }
      '?' { next(st); m = ast::m_const; }
      _ { m = ast::m_imm; }
    }
    ret {ty: parse_ty(st, conv), mutbl: m};
}

fn parse_def(st: @pstate, conv: conv_did) -> ast::def_id {
    let mut def = ~[];
    while peek(st) != '|' { vec::push(def, next_byte(st)); }
    st.pos = st.pos + 1u;
    ret conv(parse_def_id(def));
}

fn parse_int(st: @pstate) -> int {
    let mut n = 0;
    loop {
        let cur = peek(st);
        if cur < '0' || cur > '9' { ret n; }
        st.pos = st.pos + 1u;
        n *= 10;
        n += (cur as int) - ('0' as int);
    };
}

fn parse_hex(st: @pstate) -> uint {
    let mut n = 0u;
    loop {
        let cur = peek(st);
        if (cur < '0' || cur > '9') && (cur < 'a' || cur > 'f') { ret n; }
        st.pos = st.pos + 1u;
        n *= 16u;
        if '0' <= cur && cur <= '9' {
            n += (cur as uint) - ('0' as uint);
        } else { n += 10u + (cur as uint) - ('a' as uint); }
    };
}

fn parse_purity(c: char) -> purity {
    alt check c {
      'u' {unsafe_fn}
      'p' {pure_fn}
      'i' {impure_fn}
      'c' {extern_fn}
    }
}

fn parse_ty_fn(st: @pstate, conv: conv_did) -> ty::fn_ty {
    let proto = parse_proto(next(st));
    let purity = parse_purity(next(st));
    assert (next(st) == '[');
    let mut inputs: ~[ty::arg] = ~[];
    while peek(st) != ']' {
        let mode = alt check peek(st) {
          '&' { ast::by_mutbl_ref }
          '-' { ast::by_move }
          '+' { ast::by_copy }
          '=' { ast::by_ref }
          '#' { ast::by_val }
        };
        st.pos += 1u;
        vec::push(inputs, {mode: ast::expl(mode), ty: parse_ty(st, conv)});
    }
    st.pos += 1u; // eat the ']'
    let cs = parse_constrs(st, conv);
    let (ret_style, ret_ty) = parse_ret_ty(st, conv);
    ret {purity: purity, proto: proto, inputs: inputs, output: ret_ty,
         ret_style: ret_style, constraints: cs};
}


// Rust metadata parsing
fn parse_def_id(buf: &[u8]) -> ast::def_id {
    let mut colon_idx = 0u;
    let len = vec::len(buf);
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1u; }
    if colon_idx == len {
        #error("didn't find ':' when parsing def id");
        fail;
    }
    let crate_part = vec::slice(buf, 0u, colon_idx);
    let def_part = vec::slice(buf, colon_idx + 1u, len);

    let crate_num = alt uint::parse_buf(crate_part, 10u) {
       some(cn) { cn as int }
       none { fail (#fmt("internal error: parse_def_id: crate number \
         expected, but found %?", crate_part)); }
    };
    let def_num = alt uint::parse_buf(def_part, 10u) {
       some(dn) { dn as int }
       none { fail (#fmt("internal error: parse_def_id: id expected, but \
         found %?", def_part)); }
    };
    ret {crate: crate_num, node: def_num};
}

fn parse_bounds_data(data: @~[u8], start: uint,
                     crate_num: int, tcx: ty::ctxt, conv: conv_did)
    -> @~[ty::param_bound] {
    let st = @{data: data, crate: crate_num, mut pos: start, tcx: tcx};
    parse_bounds(st, conv)
}

fn parse_bounds(st: @pstate, conv: conv_did) -> @~[ty::param_bound] {
    let mut bounds = ~[];
    loop {
        vec::push(bounds, alt check next(st) {
          'S' { ty::bound_send }
          'C' { ty::bound_copy }
          'K' { ty::bound_const }
          'I' { ty::bound_trait(parse_ty(st, conv)) }
          '.' { break; }
        });
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
