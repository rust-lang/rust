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


use middle::ty;

use std::str;
use std::uint;
use syntax::abi::AbiSet;
use syntax::abi;
use syntax::ast;
use syntax::ast::*;
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
    TypeParameter,

    // Identifies a region parameter (`fn foo<'X>() { ... }`).
    RegionParameter,
}
type conv_did<'a> =
    'a |source: DefIdSource, ast::DefId| -> ast::DefId;

pub struct PState<'a> {
    data: &'a [u8],
    crate: ast::CrateNum,
    pos: uint,
    tcx: ty::ctxt
}

fn peek(st: &PState) -> char {
    st.data[st.pos] as char
}

fn next(st: &mut PState) -> char {
    let ch = st.data[st.pos] as char;
    st.pos = st.pos + 1u;
    return ch;
}

fn next_byte(st: &mut PState) -> u8 {
    let b = st.data[st.pos];
    st.pos = st.pos + 1u;
    return b;
}

fn scan<R>(st: &mut PState, is_last: |char| -> bool, op: |&[u8]| -> R) -> R {
    let start_pos = st.pos;
    debug!("scan: '{}' (start)", st.data[st.pos] as char);
    while !is_last(st.data[st.pos] as char) {
        st.pos += 1;
        debug!("scan: '{}'", st.data[st.pos] as char);
    }
    let end_pos = st.pos;
    st.pos += 1;
    return op(st.data.slice(start_pos, end_pos));
}

pub fn parse_ident(st: &mut PState, last: char) -> ast::Ident {
    fn is_last(b: char, c: char) -> bool { return c == b; }
    return parse_ident_(st, |a| is_last(last, a) );
}

fn parse_ident_(st: &mut PState, is_last: |char| -> bool) -> ast::Ident {
    scan(st, is_last, |bytes| {
            st.tcx.sess.ident_of(str::from_utf8(bytes).unwrap())
        })
}

pub fn parse_state_from_data<'a>(data: &'a [u8], crate_num: ast::CrateNum,
                             pos: uint, tcx: ty::ctxt) -> PState<'a> {
    PState {
        data: data,
        crate: crate_num,
        pos: pos,
        tcx: tcx
    }
}

pub fn parse_ty_data(data: &[u8], crate_num: ast::CrateNum, pos: uint, tcx: ty::ctxt,
                     conv: conv_did) -> ty::t {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_ty(&mut st, conv)
}

pub fn parse_bare_fn_ty_data(data: &[u8], crate_num: ast::CrateNum, pos: uint, tcx: ty::ctxt,
                             conv: conv_did) -> ty::BareFnTy {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_bare_fn_ty(&mut st, conv)
}

pub fn parse_trait_ref_data(data: &[u8], crate_num: ast::CrateNum, pos: uint, tcx: ty::ctxt,
                            conv: conv_did) -> ty::TraitRef {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_trait_ref(&mut st, conv)
}

pub fn parse_substs_data(data: &[u8], crate_num: ast::CrateNum, pos: uint, tcx: ty::ctxt,
                         conv: conv_did) -> ty::substs {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_substs(&mut st, conv)
}

fn parse_sigil(st: &mut PState) -> ast::Sigil {
    match next(st) {
        '@' => ast::ManagedSigil,
        '~' => ast::OwnedSigil,
        '&' => ast::BorrowedSigil,
        c => st.tcx.sess.bug(format!("parse_sigil(): bad input '{}'", c))
    }
}

fn parse_vstore(st: &mut PState, conv: conv_did) -> ty::vstore {
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
      '&' => ty::vstore_slice(parse_region(st, conv)),
      c => st.tcx.sess.bug(format!("parse_vstore(): bad input '{}'", c))
    }
}

fn parse_trait_store(st: &mut PState, conv: conv_did) -> ty::TraitStore {
    match next(st) {
        '~' => ty::UniqTraitStore,
        '@' => ty::BoxTraitStore,
        '&' => ty::RegionTraitStore(parse_region(st, conv)),
        c => st.tcx.sess.bug(format!("parse_trait_store(): bad input '{}'", c))
    }
}

fn parse_substs(st: &mut PState, conv: conv_did) -> ty::substs {
    let regions = parse_region_substs(st, |x,y| conv(x,y));

    let self_ty = parse_opt(st, |st| parse_ty(st, |x,y| conv(x,y)) );

    assert_eq!(next(st), '[');
    let mut params: ~[ty::t] = ~[];
    while peek(st) != ']' { params.push(parse_ty(st, |x,y| conv(x,y))); }
    st.pos = st.pos + 1u;

    return ty::substs {
        regions: regions,
        self_ty: self_ty,
        tps: params
    };
}

fn parse_region_substs(st: &mut PState, conv: conv_did) -> ty::RegionSubsts {
    match next(st) {
        'e' => ty::ErasedRegions,
        'n' => {
            let mut regions = opt_vec::Empty;
            while peek(st) != '.' {
                let r = parse_region(st, |x,y| conv(x,y));
                regions.push(r);
            }
            assert_eq!(next(st), '.');
            ty::NonerasedRegions(regions)
        }
        _ => fail!("parse_bound_region: bad input")
    }
}

fn parse_bound_region(st: &mut PState, conv: conv_did) -> ty::BoundRegion {
    match next(st) {
        'a' => {
            let id = parse_uint(st);
            assert_eq!(next(st), '|');
            ty::BrAnon(id)
        }
        '[' => {
            let def = parse_def(st, RegionParameter, |x,y| conv(x,y));
            let ident = st.tcx.sess.ident_of(parse_str(st, ']'));
            ty::BrNamed(def, ident)
        }
        'f' => {
            let id = parse_uint(st);
            assert_eq!(next(st), '|');
            ty::BrFresh(id)
        }
        _ => fail!("parse_bound_region: bad input")
    }
}

fn parse_region(st: &mut PState, conv: conv_did) -> ty::Region {
    match next(st) {
      'b' => {
        assert_eq!(next(st), '[');
        let id = parse_uint(st) as ast::NodeId;
        assert_eq!(next(st), '|');
        let br = parse_bound_region(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        ty::ReLateBound(id, br)
      }
      'B' => {
        assert_eq!(next(st), '[');
        let node_id = parse_uint(st) as ast::NodeId;
        assert_eq!(next(st), '|');
        let index = parse_uint(st);
        assert_eq!(next(st), '|');
        let nm = st.tcx.sess.ident_of(parse_str(st, ']'));
        ty::ReEarlyBound(node_id, index, nm)
      }
      'f' => {
        assert_eq!(next(st), '[');
        let id = parse_uint(st) as ast::NodeId;
        assert_eq!(next(st), '|');
        let br = parse_bound_region(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        ty::ReFree(ty::FreeRegion {scope_id: id,
                                    bound_region: br})
      }
      's' => {
        let id = parse_uint(st) as ast::NodeId;
        assert_eq!(next(st), '|');
        ty::ReScope(id)
      }
      't' => {
        ty::ReStatic
      }
      'e' => {
        ty::ReStatic
      }
      _ => fail!("parse_region: bad input")
    }
}

fn parse_opt<T>(st: &mut PState, f: |&mut PState| -> T) -> Option<T> {
    match next(st) {
      'n' => None,
      's' => Some(f(st)),
      _ => fail!("parse_opt: bad input")
    }
}

fn parse_str(st: &mut PState, term: char) -> ~str {
    let mut result = ~"";
    while peek(st) != term {
        unsafe {
            str::raw::push_byte(&mut result, next_byte(st));
        }
    }
    next(st);
    return result;
}

fn parse_trait_ref(st: &mut PState, conv: conv_did) -> ty::TraitRef {
    let def = parse_def(st, NominalType, |x,y| conv(x,y));
    let substs = parse_substs(st, |x,y| conv(x,y));
    ty::TraitRef {def_id: def, substs: substs}
}

fn parse_ty(st: &mut PState, conv: conv_did) -> ty::t {
    match next(st) {
      'n' => return ty::mk_nil(),
      'z' => return ty::mk_bot(),
      'b' => return ty::mk_bool(),
      'i' => return ty::mk_int(),
      'u' => return ty::mk_uint(),
      'M' => {
        match next(st) {
          'b' => return ty::mk_mach_uint(ast::TyU8),
          'w' => return ty::mk_mach_uint(ast::TyU16),
          'l' => return ty::mk_mach_uint(ast::TyU32),
          'd' => return ty::mk_mach_uint(ast::TyU64),
          'B' => return ty::mk_mach_int(ast::TyI8),
          'W' => return ty::mk_mach_int(ast::TyI16),
          'L' => return ty::mk_mach_int(ast::TyI32),
          'D' => return ty::mk_mach_int(ast::TyI64),
          'f' => return ty::mk_mach_float(ast::TyF32),
          'F' => return ty::mk_mach_float(ast::TyF64),
          _ => fail!("parse_ty: bad numeric type")
        }
      }
      'c' => return ty::mk_char(),
      't' => {
        assert_eq!(next(st), '[');
        let def = parse_def(st, NominalType, |x,y| conv(x,y));
        let substs = parse_substs(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        return ty::mk_enum(st.tcx, def, substs);
      }
      'x' => {
        assert_eq!(next(st), '[');
        let def = parse_def(st, NominalType, |x,y| conv(x,y));
        let substs = parse_substs(st, |x,y| conv(x,y));
        let store = parse_trait_store(st, |x,y| conv(x,y));
        let mt = parse_mutability(st);
        let bounds = parse_bounds(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        return ty::mk_trait(st.tcx, def, substs, store, mt, bounds.builtin_bounds);
      }
      'p' => {
        let did = parse_def(st, TypeParameter, |x,y| conv(x,y));
        debug!("parsed ty_param: did={:?}", did);
        return ty::mk_param(st.tcx, parse_uint(st), did);
      }
      's' => {
        let did = parse_def(st, TypeParameter, |x,y| conv(x,y));
        return ty::mk_self(st.tcx, did);
      }
      '@' => return ty::mk_box(st.tcx, parse_ty(st, |x,y| conv(x,y))),
      '~' => return ty::mk_uniq(st.tcx, parse_ty(st, |x,y| conv(x,y))),
      '*' => return ty::mk_ptr(st.tcx, parse_mt(st, |x,y| conv(x,y))),
      '&' => {
        let r = parse_region(st, |x,y| conv(x,y));
        let mt = parse_mt(st, |x,y| conv(x,y));
        return ty::mk_rptr(st.tcx, r, mt);
      }
      'U' => return ty::mk_unboxed_vec(st.tcx, parse_mt(st, |x,y| conv(x,y))),
      'V' => {
        let mt = parse_mt(st, |x,y| conv(x,y));
        let v = parse_vstore(st, |x,y| conv(x,y));
        return ty::mk_vec(st.tcx, mt, v);
      }
      'v' => {
        let v = parse_vstore(st, |x,y| conv(x,y));
        return ty::mk_str(st.tcx, v);
      }
      'T' => {
        assert_eq!(next(st), '[');
        let mut params = ~[];
        while peek(st) != ']' { params.push(parse_ty(st, |x,y| conv(x,y))); }
        st.pos = st.pos + 1u;
        return ty::mk_tup(st.tcx, params);
      }
      'f' => {
        return ty::mk_closure(st.tcx, parse_closure_ty(st, |x,y| conv(x,y)));
      }
      'F' => {
        return ty::mk_bare_fn(st.tcx, parse_bare_fn_ty(st, |x,y| conv(x,y)));
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

        let tt_opt = {
            let rcache = st.tcx.rcache.borrow();
            rcache.get().find_copy(&key)
        };
        match tt_opt {
          Some(tt) => return tt,
          None => {
            let mut ps = PState {
                pos: pos,
                .. *st
            };
            let tt = parse_ty(&mut ps, |x,y| conv(x,y));
            let mut rcache = st.tcx.rcache.borrow_mut();
            rcache.get().insert(key, tt);
            return tt;
          }
        }
      }
      '"' => {
        let _ = parse_def(st, TypeWithId, |x,y| conv(x,y));
        let inner = parse_ty(st, |x,y| conv(x,y));
        inner
      }
      'a' => {
          assert_eq!(next(st), '[');
          let did = parse_def(st, NominalType, |x,y| conv(x,y));
          let substs = parse_substs(st, |x,y| conv(x,y));
          assert_eq!(next(st), ']');
          return ty::mk_struct(st.tcx, did, substs);
      }
      c => { error!("unexpected char in type string: {}", c); fail!();}
    }
}

fn parse_mutability(st: &mut PState) -> ast::Mutability {
    match peek(st) {
      'm' => { next(st); ast::MutMutable }
      _ => { ast::MutImmutable }
    }
}

fn parse_mt(st: &mut PState, conv: conv_did) -> ty::mt {
    let m = parse_mutability(st);
    ty::mt { ty: parse_ty(st, |x,y| conv(x,y)), mutbl: m }
}

fn parse_def(st: &mut PState, source: DefIdSource,
             conv: conv_did) -> ast::DefId {
    return conv(source, scan(st, |c| { c == '|' }, parse_def_id));
}

fn parse_uint(st: &mut PState) -> uint {
    let mut n = 0;
    loop {
        let cur = peek(st);
        if cur < '0' || cur > '9' { return n; }
        st.pos = st.pos + 1u;
        n *= 10;
        n += (cur as uint) - ('0' as uint);
    };
}

fn parse_hex(st: &mut PState) -> uint {
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

fn parse_purity(c: char) -> Purity {
    match c {
        'u' => UnsafeFn,
        'i' => ImpureFn,
        'c' => ExternFn,
        _ => fail!("parse_purity: bad purity {}", c)
    }
}

fn parse_abi_set(st: &mut PState) -> AbiSet {
    assert_eq!(next(st), '[');
    let mut abis = AbiSet::empty();
    while peek(st) != ']' {
         scan(st, |c| c == ',', |bytes| {
                 let abi_str = str::from_utf8(bytes).unwrap().to_owned();
                 let abi = abi::lookup(abi_str).expect(abi_str);
                 abis.add(abi);
              });
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

fn parse_closure_ty(st: &mut PState, conv: conv_did) -> ty::ClosureTy {
    let sigil = parse_sigil(st);
    let purity = parse_purity(next(st));
    let onceness = parse_onceness(next(st));
    let region = parse_region(st, |x,y| conv(x,y));
    let bounds = parse_bounds(st, |x,y| conv(x,y));
    let sig = parse_sig(st, |x,y| conv(x,y));
    ty::ClosureTy {
        purity: purity,
        sigil: sigil,
        onceness: onceness,
        region: region,
        bounds: bounds.builtin_bounds,
        sig: sig
    }
}

fn parse_bare_fn_ty(st: &mut PState, conv: conv_did) -> ty::BareFnTy {
    let purity = parse_purity(next(st));
    let abi = parse_abi_set(st);
    let sig = parse_sig(st, |x,y| conv(x,y));
    ty::BareFnTy {
        purity: purity,
        abis: abi,
        sig: sig
    }
}

fn parse_sig(st: &mut PState, conv: conv_did) -> ty::FnSig {
    assert_eq!(next(st), '[');
    let id = parse_uint(st) as ast::NodeId;
    assert_eq!(next(st), '|');
    let mut inputs = ~[];
    while peek(st) != ']' {
        inputs.push(parse_ty(st, |x,y| conv(x,y)));
    }
    st.pos += 1u; // eat the ']'
    let variadic = match next(st) {
        'V' => true,
        'N' => false,
        r => fail!(format!("Bad variadic: {}", r)),
    };
    let ret_ty = parse_ty(st, |x,y| conv(x,y));
    ty::FnSig {binder_id: id,
               inputs: inputs,
               output: ret_ty,
               variadic: variadic}
}

// Rust metadata parsing
pub fn parse_def_id(buf: &[u8]) -> ast::DefId {
    let mut colon_idx = 0u;
    let len = buf.len();
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1u; }
    if colon_idx == len {
        error!("didn't find ':' when parsing def id");
        fail!();
    }

    let crate_part = buf.slice(0u, colon_idx);
    let def_part = buf.slice(colon_idx + 1u, len);

    let crate_num = match uint::parse_bytes(crate_part, 10u) {
       Some(cn) => cn as ast::CrateNum,
       None => fail!("internal error: parse_def_id: crate number expected, but found {:?}",
                     crate_part)
    };
    let def_num = match uint::parse_bytes(def_part, 10u) {
       Some(dn) => dn as ast::NodeId,
       None => fail!("internal error: parse_def_id: id expected, but found {:?}",
                     def_part)
    };
    ast::DefId { crate: crate_num, node: def_num }
}

pub fn parse_type_param_def_data(data: &[u8], start: uint,
                                 crate_num: ast::CrateNum, tcx: ty::ctxt,
                                 conv: conv_did) -> ty::TypeParameterDef
{
    let mut st = parse_state_from_data(data, crate_num, start, tcx);
    parse_type_param_def(&mut st, conv)
}

fn parse_type_param_def(st: &mut PState, conv: conv_did) -> ty::TypeParameterDef {
    ty::TypeParameterDef {ident: parse_ident(st, ':'),
                          def_id: parse_def(st, NominalType, |x,y| conv(x,y)),
                          bounds: @parse_bounds(st, |x,y| conv(x,y))}
}

fn parse_bounds(st: &mut PState, conv: conv_did) -> ty::ParamBounds {
    let mut param_bounds = ty::ParamBounds {
        builtin_bounds: ty::EmptyBuiltinBounds(),
        trait_bounds: ~[]
    };
    loop {
        match next(st) {
            'S' => {
                param_bounds.builtin_bounds.add(ty::BoundSend);
            }
            'K' => {
                param_bounds.builtin_bounds.add(ty::BoundFreeze);
            }
            'O' => {
                param_bounds.builtin_bounds.add(ty::BoundStatic);
            }
            'Z' => {
                param_bounds.builtin_bounds.add(ty::BoundSized);
            }
            'P' => {
                param_bounds.builtin_bounds.add(ty::BoundPod);
            }
            'I' => {
                param_bounds.trait_bounds.push(@parse_trait_ref(st, |x,y| conv(x,y)));
            }
            '.' => {
                return param_bounds;
            }
            c => {
                fail!("parse_bounds: bad bounds ('{}')", c)
            }
        }
    }
}
