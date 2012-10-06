// Type decoding

// tjc note: Would be great to have a `match check` macro equivalent
// for some of these

use syntax::ast;
use syntax::ast::*;
use syntax::ast_util;
use syntax::ast_util::respan;
use middle::ty;
use std::map::HashMap;
use ty::{FnTyBase, FnMeta, FnSig};

export parse_state_from_data;
export parse_arg_data, parse_ty_data, parse_def_id, parse_ident;
export parse_bounds_data;
export pstate;

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
    return ch;
}

fn next_byte(st: @pstate) -> u8 {
    let b = st.data[st.pos];
    st.pos = st.pos + 1u;
    return b;
}

fn parse_ident(st: @pstate, last: char) -> ast::ident {
    fn is_last(b: char, c: char) -> bool { return c == b; }
    return parse_ident_(st, |a| is_last(last, a) );
}

fn parse_ident_(st: @pstate, is_last: fn@(char) -> bool) ->
   ast::ident {
    let mut rslt = ~"";
    while !is_last(peek(st)) {
        rslt += str::from_byte(next_byte(st));
    }
    return st.tcx.sess.ident_of(rslt);
}

fn parse_state_from_data(data: @~[u8], crate_num: int,
                         pos: uint, tcx: ty::ctxt)
    -> @pstate
{
    @{data: data, crate: crate_num, mut pos: pos, tcx: tcx}
}

fn parse_ty_data(data: @~[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                 conv: conv_did) -> ty::t {
    let st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_ty(st, conv)
}

fn parse_arg_data(data: @~[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                  conv: conv_did) -> ty::arg {
    let st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_arg(st, conv)
}

fn parse_ret_ty(st: @pstate, conv: conv_did) -> (ast::ret_style, ty::t) {
    match peek(st) {
      '!' => { next(st); (ast::noreturn, ty::mk_bot(st.tcx)) }
      _ => (ast::return_val, parse_ty(st, conv))
    }
}

fn parse_path(st: @pstate) -> @ast::path {
    let mut idents: ~[ast::ident] = ~[];
    fn is_last(c: char) -> bool { return c == '(' || c == ':'; }
    idents.push(parse_ident_(st, is_last));
    loop {
        match peek(st) {
          ':' => { next(st); next(st); }
          c => {
            if c == '(' {
                return @{span: ast_util::dummy_sp(),
                      global: false, idents: idents,
                      rp: None, types: ~[]};
            } else { idents.push(parse_ident_(st, is_last)); }
          }
        }
    };
}

fn parse_ty_rust_fn(st: @pstate, conv: conv_did) -> ty::t {
    return ty::mk_fn(st.tcx, parse_ty_fn(st, conv));
}

fn parse_proto(st: @pstate) -> ty::fn_proto {
    match next(st) {
        'n' => ty::proto_bare,
        'v' => ty::proto_vstore(parse_vstore(st)),
        c => fail ~"illegal proto type kind " + str::from_char(c)
    }
}

fn parse_vstore(st: @pstate) -> ty::vstore {
    assert next(st) == '/';

    let c = peek(st);
    if '0' <= c && c <= '9' {
        let n = parse_int(st) as uint;
        assert next(st) == '|';
        return ty::vstore_fixed(n);
    }

    match next(st) {
      '~' => ty::vstore_uniq,
      '@' => ty::vstore_box,
      '&' => ty::vstore_slice(parse_region(st)),
      _ => fail ~"parse_vstore: bad input"
    }
}

fn parse_substs(st: @pstate, conv: conv_did) -> ty::substs {
    let self_r = parse_opt(st, || parse_region(st) );

    let self_ty = parse_opt(st, || parse_ty(st, conv) );

    assert next(st) == '[';
    let mut params: ~[ty::t] = ~[];
    while peek(st) != ']' { params.push(parse_ty(st, conv)); }
    st.pos = st.pos + 1u;

    return {self_r: self_r,
         self_ty: self_ty,
         tps: params};
}

fn parse_bound_region(st: @pstate) -> ty::bound_region {
    match next(st) {
      's' => ty::br_self,
      'a' => {
        let id = parse_int(st) as uint;
        assert next(st) == '|';
        ty::br_anon(id)
      }
      '[' => ty::br_named(st.tcx.sess.ident_of(parse_str(st, ']'))),
      'c' => {
        let id = parse_int(st);
        assert next(st) == '|';
        ty::br_cap_avoid(id, @parse_bound_region(st))
      },
      _ => fail ~"parse_bound_region: bad input"
    }
}

fn parse_region(st: @pstate) -> ty::region {
    match next(st) {
      'b' => {
        ty::re_bound(parse_bound_region(st))
      }
      'f' => {
        assert next(st) == '[';
        let id = parse_int(st);
        assert next(st) == '|';
        let br = parse_bound_region(st);
        assert next(st) == ']';
        ty::re_free(id, br)
      }
      's' => {
        let id = parse_int(st);
        assert next(st) == '|';
        ty::re_scope(id)
      }
      't' => {
        ty::re_static
      }
      _ => fail ~"parse_region: bad input"
    }
}

fn parse_opt<T>(st: @pstate, f: fn() -> T) -> Option<T> {
    match next(st) {
      'n' => None,
      's' => Some(f()),
      _ => fail ~"parse_opt: bad input"
    }
}

fn parse_str(st: @pstate, term: char) -> ~str {
    let mut result = ~"";
    while peek(st) != term {
        result += str::from_byte(next_byte(st));
    }
    next(st);
    return result;
}

fn parse_ty(st: @pstate, conv: conv_did) -> ty::t {
    match next(st) {
      'n' => return ty::mk_nil(st.tcx),
      'z' => return ty::mk_bot(st.tcx),
      'b' => return ty::mk_bool(st.tcx),
      'i' => return ty::mk_int(st.tcx),
      'u' => return ty::mk_uint(st.tcx),
      'l' => return ty::mk_float(st.tcx),
      'M' => {
        match next(st) {
          'b' => return ty::mk_mach_uint(st.tcx, ast::ty_u8),
          'w' => return ty::mk_mach_uint(st.tcx, ast::ty_u16),
          'l' => return ty::mk_mach_uint(st.tcx, ast::ty_u32),
          'd' => return ty::mk_mach_uint(st.tcx, ast::ty_u64),
          'B' => return ty::mk_mach_int(st.tcx, ast::ty_i8),
          'W' => return ty::mk_mach_int(st.tcx, ast::ty_i16),
          'L' => return ty::mk_mach_int(st.tcx, ast::ty_i32),
          'D' => return ty::mk_mach_int(st.tcx, ast::ty_i64),
          'f' => return ty::mk_mach_float(st.tcx, ast::ty_f32),
          'F' => return ty::mk_mach_float(st.tcx, ast::ty_f64),
          _ => fail ~"parse_ty: bad numeric type"
        }
      }
      'c' => return ty::mk_char(st.tcx),
      't' => {
        assert (next(st) == '[');
        let def = parse_def(st, conv);
        let substs = parse_substs(st, conv);
        assert next(st) == ']';
        return ty::mk_enum(st.tcx, def, substs);
      }
      'x' => {
        assert next(st) == '[';
        let def = parse_def(st, conv);
        let substs = parse_substs(st, conv);
        let vstore = parse_vstore(st);
        assert next(st) == ']';
        return ty::mk_trait(st.tcx, def, substs, vstore);
      }
      'p' => {
        let did = parse_def(st, conv);
        return ty::mk_param(st.tcx, parse_int(st) as uint, did);
      }
      's' => {
        return ty::mk_self(st.tcx);
      }
      '@' => return ty::mk_box(st.tcx, parse_mt(st, conv)),
      '~' => return ty::mk_uniq(st.tcx, parse_mt(st, conv)),
      '*' => return ty::mk_ptr(st.tcx, parse_mt(st, conv)),
      '&' => {
        let r = parse_region(st);
        let mt = parse_mt(st, conv);
        return ty::mk_rptr(st.tcx, r, mt);
      }
      'U' => return ty::mk_unboxed_vec(st.tcx, parse_mt(st, conv)),
      'V' => {
        let mt = parse_mt(st, conv);
        let v = parse_vstore(st);
        return ty::mk_evec(st.tcx, mt, v);
      }
      'v' => {
        let v = parse_vstore(st);
        return ty::mk_estr(st.tcx, v);
      }
      'R' => {
        assert (next(st) == '[');
        let mut fields: ~[ty::field] = ~[];
        while peek(st) != ']' {
            let name = st.tcx.sess.ident_of(parse_str(st, '='));
            fields.push({ident: name, mt: parse_mt(st, conv)});
        }
        st.pos = st.pos + 1u;
        return ty::mk_rec(st.tcx, fields);
      }
      'T' => {
        assert (next(st) == '[');
        let mut params = ~[];
        while peek(st) != ']' { params.push(parse_ty(st, conv)); }
        st.pos = st.pos + 1u;
        return ty::mk_tup(st.tcx, params);
      }
      'f' => {
        parse_ty_rust_fn(st, conv)
      }
      'X' => {
        return ty::mk_var(st.tcx, ty::TyVid(parse_int(st) as uint));
      }
      'Y' => return ty::mk_type(st.tcx),
      'C' => {
        let ck = match next(st) {
          '&' => ty::ck_block,
          '@' => ty::ck_box,
          '~' => ty::ck_uniq,
          _ => fail ~"parse_ty: bad closure kind"
        };
        return ty::mk_opaque_closure_ptr(st.tcx, ck);
      }
      '#' => {
        let pos = parse_hex(st);
        assert (next(st) == ':');
        let len = parse_hex(st);
        assert (next(st) == '#');
        match st.tcx.rcache.find({cnum: st.crate, pos: pos, len: len}) {
          Some(tt) => return tt,
          None => {
            let ps = @{pos: pos ,.. *st};
            let tt = parse_ty(ps, conv);
            st.tcx.rcache.insert({cnum: st.crate, pos: pos, len: len}, tt);
            return tt;
          }
        }
      }
      '"' => {
        let def = parse_def(st, conv);
        let inner = parse_ty(st, conv);
        ty::mk_with_id(st.tcx, inner, def)
      }
      'B' => ty::mk_opaque_box(st.tcx),
      'a' => {
          debug!("saw a class");
          assert (next(st) == '[');
          debug!("saw a [");
          let did = parse_def(st, conv);
          debug!("parsed a def_id %?", did);
          let substs = parse_substs(st, conv);
          assert (next(st) == ']');
          return ty::mk_class(st.tcx, did, substs);
      }
      c => { error!("unexpected char in type string: %c", c); fail;}
    }
}

fn parse_mt(st: @pstate, conv: conv_did) -> ty::mt {
    let mut m;
    match peek(st) {
      'm' => { next(st); m = ast::m_mutbl; }
      '?' => { next(st); m = ast::m_const; }
      _ => { m = ast::m_imm; }
    }
    return {ty: parse_ty(st, conv), mutbl: m};
}

fn parse_def(st: @pstate, conv: conv_did) -> ast::def_id {
    let mut def = ~[];
    while peek(st) != '|' { def.push(next_byte(st)); }
    st.pos = st.pos + 1u;
    return conv(parse_def_id(def));
}

fn parse_int(st: @pstate) -> int {
    let mut n = 0;
    loop {
        let cur = peek(st);
        if cur < '0' || cur > '9' { return n; }
        st.pos = st.pos + 1u;
        n *= 10;
        n += (cur as int) - ('0' as int);
    };
}

fn parse_hex(st: @pstate) -> uint {
    let mut n = 0u;
    loop {
        let cur = peek(st);
        if (cur < '0' || cur > '9') && (cur < 'a' || cur > 'f') { return n; }
        st.pos = st.pos + 1u;
        n *= 16u;
        if '0' <= cur && cur <= '9' {
            n += (cur as uint) - ('0' as uint);
        } else { n += 10u + (cur as uint) - ('a' as uint); }
    };
}

fn parse_purity(c: char) -> purity {
    match c {
      'u' => unsafe_fn,
      'p' => pure_fn,
      'i' => impure_fn,
      'c' => extern_fn,
      _ => fail ~"parse_purity: bad purity"
    }
}

fn parse_arg(st: @pstate, conv: conv_did) -> ty::arg {
    {mode: parse_mode(st),
     ty: parse_ty(st, conv)}
}

fn parse_mode(st: @pstate) -> ast::mode {
    let m = ast::expl(match next(st) {
        '-' => ast::by_move,
        '+' => ast::by_copy,
        '=' => ast::by_ref,
        '#' => ast::by_val,
        _ => fail ~"bad mode"
    });
    return m;
}

fn parse_ty_fn(st: @pstate, conv: conv_did) -> ty::FnTy {
    let proto = parse_proto(st);
    let purity = parse_purity(next(st));
    let bounds = parse_bounds(st, conv);
    assert (next(st) == '[');
    let mut inputs: ~[ty::arg] = ~[];
    while peek(st) != ']' {
        let mode = parse_mode(st);
        inputs.push({mode: mode, ty: parse_ty(st, conv)});
    }
    st.pos += 1u; // eat the ']'
    let (ret_style, ret_ty) = parse_ret_ty(st, conv);
    return FnTyBase {
        meta: FnMeta {purity: purity,
                      proto: proto,
                      bounds: bounds,
                      ret_style: ret_style},
        sig: FnSig {inputs: inputs,
                    output: ret_ty}
    };
}


// Rust metadata parsing
fn parse_def_id(buf: &[u8]) -> ast::def_id {
    let mut colon_idx = 0u;
    let len = vec::len(buf);
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1u; }
    if colon_idx == len {
        error!("didn't find ':' when parsing def id");
        fail;
    }

    let crate_part = vec::view(buf, 0u, colon_idx);
    let def_part = vec::view(buf, colon_idx + 1u, len);

    let crate_num = match uint::parse_bytes(crate_part, 10u) {
       Some(cn) => cn as int,
       None => fail (fmt!("internal error: parse_def_id: crate number \
                               expected, but found %?", crate_part))
    };
    let def_num = match uint::parse_bytes(def_part, 10u) {
       Some(dn) => dn as int,
       None => fail (fmt!("internal error: parse_def_id: id expected, but \
                               found %?", def_part))
    };
    return {crate: crate_num, node: def_num};
}

fn parse_bounds_data(data: @~[u8], start: uint,
                     crate_num: int, tcx: ty::ctxt, conv: conv_did)
    -> @~[ty::param_bound]
{
    let st = parse_state_from_data(data, crate_num, start, tcx);
    parse_bounds(st, conv)
}

fn parse_bounds(st: @pstate, conv: conv_did) -> @~[ty::param_bound] {
    let mut bounds = ~[];
    loop {
        bounds.push(match next(st) {
          'S' => ty::bound_send,
          'C' => ty::bound_copy,
          'K' => ty::bound_const,
          'O' => ty::bound_owned,
          'I' => ty::bound_trait(parse_ty(st, conv)),
          '.' => break,
          _ => fail ~"parse_bounds: bad bounds"
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
