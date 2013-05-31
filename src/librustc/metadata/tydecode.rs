// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Type decoding

// tjc note: Would be great to have a `match check` macro equivalent
// for some of these

use core::prelude::*;

use middle::ty;

use core::str;
use core::uint;
use core::vec;
use syntax::abi::AbiSet;
use syntax::abi;
use syntax::ast;
use syntax::ast::*;
use syntax::codemap::dummy_sp;
use syntax::opt_vec;

// Compact string representation for ty::t values. API ty_str &
// parse_from_str. Extra parameters are for converting to/from def_ids in the
// data buffer. Whatever format you choose should not contain pipe characters.

// Def id conversion: when we encounter def-ids, they have to be translated.
// For example, the crate number must be converted from the crate number used
// in the library we are reading from into the local crate numbers in use
// here.  To perform this translation, the type decoder is supplied with a
// conversion function of type `conv_did`.
//
// Sometimes, particularly when inlining, the correct translation of the
// def-id will depend on where it originated from.  Therefore, the conversion
// function is given an indicator of the source of the def-id.  See
// astencode.rs for more information.
pub enum DefIdSource {
    // Identifies a struct, trait, enum, etc.
    NominalType,

    // Identifies a type alias (`type X = ...`).
    TypeWithId,

    // Identifies a type parameter (`fn foo<X>() { ... }`).
    TypeParameter
}
type conv_did<'self> =
    &'self fn(source: DefIdSource, ast::def_id) -> ast::def_id;

pub struct PState {
    data: @~[u8],
    crate: int,
    pos: uint,
    tcx: ty::ctxt
}

fn peek(st: @mut PState) -> char {
    st.data[st.pos] as char
}

fn next(st: @mut PState) -> char {
    let ch = st.data[st.pos] as char;
    st.pos = st.pos + 1u;
    return ch;
}

fn next_byte(st: @mut PState) -> u8 {
    let b = st.data[st.pos];
    st.pos = st.pos + 1u;
    return b;
}

fn scan<R>(st: &mut PState, is_last: &fn(char) -> bool,
           op: &fn(&[u8]) -> R) -> R
{
    let start_pos = st.pos;
    debug!("scan: '%c' (start)", st.data[st.pos] as char);
    while !is_last(st.data[st.pos] as char) {
        st.pos += 1;
        debug!("scan: '%c'", st.data[st.pos] as char);
    }
    let end_pos = st.pos;
    st.pos += 1;
    return op(st.data.slice(start_pos, end_pos));
}

pub fn parse_ident(st: @mut PState, last: char) -> ast::ident {
    fn is_last(b: char, c: char) -> bool { return c == b; }
    return parse_ident_(st, |a| is_last(last, a) );
}

fn parse_ident_(st: @mut PState, is_last: @fn(char) -> bool) ->
   ast::ident {
    let rslt = scan(st, is_last, str::from_bytes);
    return st.tcx.sess.ident_of(rslt);
}

pub fn parse_state_from_data(data: @~[u8], crate_num: int,
                             pos: uint, tcx: ty::ctxt) -> @mut PState {
    @mut PState {
        data: data,
        crate: crate_num,
        pos: pos,
        tcx: tcx
    }
}

pub fn parse_ty_data(data: @~[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                     conv: conv_did) -> ty::t {
    let st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_ty(st, conv)
}

pub fn parse_bare_fn_ty_data(data: @~[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                             conv: conv_did) -> ty::BareFnTy {
    let st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_bare_fn_ty(st, conv)
}

pub fn parse_trait_ref_data(data: @~[u8], crate_num: int, pos: uint, tcx: ty::ctxt,
                            conv: conv_did) -> ty::TraitRef {
    let st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_trait_ref(st, conv)
}

fn parse_path(st: @mut PState) -> @ast::Path {
    let mut idents: ~[ast::ident] = ~[];
    fn is_last(c: char) -> bool { return c == '(' || c == ':'; }
    idents.push(parse_ident_(st, is_last));
    loop {
        match peek(st) {
          ':' => { next(st); next(st); }
          c => {
            if c == '(' {
                return @ast::Path { span: dummy_sp(),
                                    global: false,
                                    idents: idents,
                                    rp: None,
                                    types: ~[] };
            } else { idents.push(parse_ident_(st, is_last)); }
          }
        }
    };
}

fn parse_sigil(st: @mut PState) -> ast::Sigil {
    match next(st) {
        '@' => ast::ManagedSigil,
        '~' => ast::OwnedSigil,
        '&' => ast::BorrowedSigil,
        c => st.tcx.sess.bug(fmt!("parse_sigil(): bad input '%c'", c))
    }
}

fn parse_vstore(st: @mut PState) -> ty::vstore {
    assert_eq!(next(st), '/');

    let c = peek(st);
    if '0' <= c && c <= '9' {
        let n = parse_uint(st);
        assert_eq!(next(st), '|');
        return ty::vstore_fixed(n);
    }

    match next(st) {
      '~' => ty::vstore_uniq,
      '@' => ty::vstore_box,
      '&' => ty::vstore_slice(parse_region(st)),
      c => st.tcx.sess.bug(fmt!("parse_vstore(): bad input '%c'", c))
    }
}

fn parse_trait_store(st: @mut PState) -> ty::TraitStore {
    match next(st) {
        '~' => ty::UniqTraitStore,
        '@' => ty::BoxTraitStore,
        '&' => ty::RegionTraitStore(parse_region(st)),
        c => st.tcx.sess.bug(fmt!("parse_trait_store(): bad input '%c'", c))
    }
}

fn parse_substs(st: @mut PState, conv: conv_did) -> ty::substs {
    let self_r = parse_opt(st, || parse_region(st) );

    let self_ty = parse_opt(st, || parse_ty(st, conv) );

    assert_eq!(next(st), '[');
    let mut params: ~[ty::t] = ~[];
    while peek(st) != ']' { params.push(parse_ty(st, conv)); }
    st.pos = st.pos + 1u;

    return ty::substs {
        self_r: self_r,
        self_ty: self_ty,
        tps: params
    };
}

fn parse_bound_region(st: @mut PState) -> ty::bound_region {
    match next(st) {
      's' => ty::br_self,
      'a' => {
        let id = parse_uint(st);
        assert_eq!(next(st), '|');
        ty::br_anon(id)
      }
      '[' => ty::br_named(st.tcx.sess.ident_of(parse_str(st, ']'))),
      'c' => {
        let id = parse_uint(st) as int;
        assert_eq!(next(st), '|');
        ty::br_cap_avoid(id, @parse_bound_region(st))
      },
      _ => fail!("parse_bound_region: bad input")
    }
}

fn parse_region(st: @mut PState) -> ty::Region {
    match next(st) {
      'b' => {
        ty::re_bound(parse_bound_region(st))
      }
      'f' => {
        assert_eq!(next(st), '[');
        let id = parse_uint(st) as int;
        assert_eq!(next(st), '|');
        let br = parse_bound_region(st);
        assert_eq!(next(st), ']');
        ty::re_free(ty::FreeRegion {scope_id: id,
                                    bound_region: br})
      }
      's' => {
        let id = parse_uint(st) as int;
        assert_eq!(next(st), '|');
        ty::re_scope(id)
      }
      't' => {
        ty::re_static
      }
      'e' => {
        ty::re_static
      }
      _ => fail!("parse_region: bad input")
    }
}

fn parse_opt<T>(st: @mut PState, f: &fn() -> T) -> Option<T> {
    match next(st) {
      'n' => None,
      's' => Some(f()),
      _ => fail!("parse_opt: bad input")
    }
}

fn parse_str(st: @mut PState, term: char) -> ~str {
    let mut result = ~"";
    while peek(st) != term {
        result += str::from_byte(next_byte(st));
    }
    next(st);
    return result;
}

fn parse_trait_ref(st: @mut PState, conv: conv_did) -> ty::TraitRef {
    let def = parse_def(st, NominalType, conv);
    let substs = parse_substs(st, conv);
    ty::TraitRef {def_id: def, substs: substs}
}

fn parse_ty(st: @mut PState, conv: conv_did) -> ty::t {
    match next(st) {
      'n' => return ty::mk_nil(),
      'z' => return ty::mk_bot(),
      'b' => return ty::mk_bool(),
      'i' => return ty::mk_int(),
      'u' => return ty::mk_uint(),
      'l' => return ty::mk_float(),
      'M' => {
        match next(st) {
          'b' => return ty::mk_mach_uint(ast::ty_u8),
          'w' => return ty::mk_mach_uint(ast::ty_u16),
          'l' => return ty::mk_mach_uint(ast::ty_u32),
          'd' => return ty::mk_mach_uint(ast::ty_u64),
          'B' => return ty::mk_mach_int(ast::ty_i8),
          'W' => return ty::mk_mach_int(ast::ty_i16),
          'L' => return ty::mk_mach_int(ast::ty_i32),
          'D' => return ty::mk_mach_int(ast::ty_i64),
          'f' => return ty::mk_mach_float(ast::ty_f32),
          'F' => return ty::mk_mach_float(ast::ty_f64),
          _ => fail!("parse_ty: bad numeric type")
        }
      }
      'c' => return ty::mk_char(),
      't' => {
        assert_eq!(next(st), '[');
        let def = parse_def(st, NominalType, conv);
        let substs = parse_substs(st, conv);
        assert_eq!(next(st), ']');
        return ty::mk_enum(st.tcx, def, substs);
      }
      'x' => {
        assert_eq!(next(st), '[');
        let def = parse_def(st, NominalType, conv);
        let substs = parse_substs(st, conv);
        let store = parse_trait_store(st);
        let mt = parse_mutability(st);
        assert_eq!(next(st), ']');
        return ty::mk_trait(st.tcx, def, substs, store, mt);
      }
      'p' => {
        let did = parse_def(st, TypeParameter, conv);
        debug!("parsed ty_param: did=%?", did);
        return ty::mk_param(st.tcx, parse_uint(st), did);
      }
      's' => {
        let did = parse_def(st, TypeParameter, conv);
        return ty::mk_self(st.tcx, did);
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
      'T' => {
        assert_eq!(next(st), '[');
        let mut params = ~[];
        while peek(st) != ']' { params.push(parse_ty(st, conv)); }
        st.pos = st.pos + 1u;
        return ty::mk_tup(st.tcx, params);
      }
      'f' => {
        return ty::mk_closure(st.tcx, parse_closure_ty(st, conv));
      }
      'F' => {
        return ty::mk_bare_fn(st.tcx, parse_bare_fn_ty(st, conv));
      }
      'Y' => return ty::mk_type(st.tcx),
      'C' => {
        let sigil = parse_sigil(st);
        return ty::mk_opaque_closure_ptr(st.tcx, sigil);
      }
      '#' => {
        let pos = parse_hex(st);
        assert_eq!(next(st), ':');
        let len = parse_hex(st);
        assert_eq!(next(st), '#');
        let key = ty::creader_cache_key {cnum: st.crate,
                                         pos: pos,
                                         len: len };
        match st.tcx.rcache.find(&key) {
          Some(&tt) => return tt,
          None => {
            let ps = @mut PState {pos: pos ,.. copy *st};
            let tt = parse_ty(ps, conv);
            st.tcx.rcache.insert(key, tt);
            return tt;
          }
        }
      }
      '"' => {
        let _ = parse_def(st, TypeWithId, conv);
        let inner = parse_ty(st, conv);
        inner
      }
      'B' => ty::mk_opaque_box(st.tcx),
      'a' => {
          assert_eq!(next(st), '[');
          let did = parse_def(st, NominalType, conv);
          let substs = parse_substs(st, conv);
          assert_eq!(next(st), ']');
          return ty::mk_struct(st.tcx, did, substs);
      }
      c => { error!("unexpected char in type string: %c", c); fail!();}
    }
}

fn parse_mutability(st: @mut PState) -> ast::mutability {
    match peek(st) {
      'm' => { next(st); ast::m_mutbl }
      '?' => { next(st); ast::m_const }
      _ => { ast::m_imm }
    }
}

fn parse_mt(st: @mut PState, conv: conv_did) -> ty::mt {
    let m = parse_mutability(st);
    ty::mt { ty: parse_ty(st, conv), mutbl: m }
}

fn parse_def(st: @mut PState, source: DefIdSource,
             conv: conv_did) -> ast::def_id {
    let mut def = ~[];
    while peek(st) != '|' { def.push(next_byte(st)); }
    st.pos = st.pos + 1u;
    return conv(source, parse_def_id(def));
}

fn parse_uint(st: @mut PState) -> uint {
    let mut n = 0;
    loop {
        let cur = peek(st);
        if cur < '0' || cur > '9' { return n; }
        st.pos = st.pos + 1u;
        n *= 10;
        n += (cur as uint) - ('0' as uint);
    };
}

fn parse_hex(st: @mut PState) -> uint {
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
      _ => fail!("parse_purity: bad purity")
    }
}

fn parse_abi_set(st: @mut PState) -> AbiSet {
    assert_eq!(next(st), '[');
    let mut abis = AbiSet::empty();
    while peek(st) != ']' {
        // FIXME(#5422) str API should not force this copy
        let abi_str = scan(st, |c| c == ',', str::from_bytes);
        let abi = abi::lookup(abi_str).expect(abi_str);
        abis.add(abi);
    }
    assert_eq!(next(st), ']');
    return abis;
}

fn parse_onceness(c: char) -> ast::Onceness {
    match c {
        'o' => ast::Once,
        'm' => ast::Many,
        _ => fail!("parse_onceness: bad onceness")
    }
}

fn parse_closure_ty(st: @mut PState, conv: conv_did) -> ty::ClosureTy {
    let sigil = parse_sigil(st);
    let purity = parse_purity(next(st));
    let onceness = parse_onceness(next(st));
    let region = parse_region(st);
    let bounds = parse_bounds(st, conv);
    let sig = parse_sig(st, conv);
    ty::ClosureTy {
        purity: purity,
        sigil: sigil,
        onceness: onceness,
        region: region,
        bounds: bounds.builtin_bounds,
        sig: sig
    }
}

fn parse_bare_fn_ty(st: @mut PState, conv: conv_did) -> ty::BareFnTy {
    let purity = parse_purity(next(st));
    let abi = parse_abi_set(st);
    let sig = parse_sig(st, conv);
    ty::BareFnTy {
        purity: purity,
        abis: abi,
        sig: sig
    }
}

fn parse_sig(st: @mut PState, conv: conv_did) -> ty::FnSig {
    assert_eq!(next(st), '[');
    let mut inputs = ~[];
    while peek(st) != ']' {
        inputs.push(parse_ty(st, conv));
    }
    st.pos += 1u; // eat the ']'
    let ret_ty = parse_ty(st, conv);
    ty::FnSig {bound_lifetime_names: opt_vec::Empty, // FIXME(#4846)
               inputs: inputs,
               output: ret_ty}
}

// Rust metadata parsing
pub fn parse_def_id(buf: &[u8]) -> ast::def_id {
    let mut colon_idx = 0u;
    let len = buf.len();
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1u; }
    if colon_idx == len {
        error!("didn't find ':' when parsing def id");
        fail!();
    }

    let crate_part = vec::slice(buf, 0u, colon_idx);
    let def_part = vec::slice(buf, colon_idx + 1u, len);

    let crate_num = match uint::parse_bytes(crate_part, 10u) {
       Some(cn) => cn as int,
       None => fail!("internal error: parse_def_id: crate number expected, but found %?",
                     crate_part)
    };
    let def_num = match uint::parse_bytes(def_part, 10u) {
       Some(dn) => dn as int,
       None => fail!("internal error: parse_def_id: id expected, but found %?",
                     def_part)
    };
    ast::def_id { crate: crate_num, node: def_num }
}

pub fn parse_type_param_def_data(data: @~[u8], start: uint,
                                 crate_num: int, tcx: ty::ctxt,
                                 conv: conv_did) -> ty::TypeParameterDef
{
    let st = parse_state_from_data(data, crate_num, start, tcx);
    parse_type_param_def(st, conv)
}

fn parse_type_param_def(st: @mut PState, conv: conv_did) -> ty::TypeParameterDef {
    ty::TypeParameterDef {def_id: parse_def(st, NominalType, conv),
                          bounds: @parse_bounds(st, conv)}
}

fn parse_bounds(st: @mut PState, conv: conv_did) -> ty::ParamBounds {
    let mut param_bounds = ty::ParamBounds {
        builtin_bounds: ty::EmptyBuiltinBounds(),
        trait_bounds: ~[]
    };
    loop {
        match next(st) {
            'S' => {
                param_bounds.builtin_bounds.add(ty::BoundOwned);
            }
            'C' => {
                param_bounds.builtin_bounds.add(ty::BoundCopy);
            }
            'K' => {
                param_bounds.builtin_bounds.add(ty::BoundConst);
            }
            'O' => {
                param_bounds.builtin_bounds.add(ty::BoundStatic);
            }
            'Z' => {
                param_bounds.builtin_bounds.add(ty::BoundSized);
            }
            'I' => {
                param_bounds.trait_bounds.push(@parse_trait_ref(st, conv));
            }
            '.' => {
                return param_bounds;
            }
            _ => {
                fail!("parse_bounds: bad bounds")
            }
        }
    }
}
