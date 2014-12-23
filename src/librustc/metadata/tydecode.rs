// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

#![allow(non_camel_case_types)]

pub use self::DefIdSource::*;

use middle::region;
use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::{mod, AsPredicate, Ty};

use std::rc::Rc;
use std::str;
use std::string::String;
use syntax::abi;
use syntax::ast;
use syntax::parse::token;

// Compact string representation for Ty values. API ty_str &
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
#[deriving(Copy, Show)]
pub enum DefIdSource {
    // Identifies a struct, trait, enum, etc.
    NominalType,

    // Identifies a type alias (`type X = ...`).
    TypeWithId,

    // Identifies a type parameter (`fn foo<X>() { ... }`).
    TypeParameter,

    // Identifies a region parameter (`fn foo<'X>() { ... }`).
    RegionParameter,

    // Identifies an unboxed closure
    UnboxedClosureSource
}

pub type conv_did<'a> =
    |source: DefIdSource, ast::DefId|: 'a -> ast::DefId;

pub struct PState<'a, 'tcx: 'a> {
    data: &'a [u8],
    krate: ast::CrateNum,
    pos: uint,
    tcx: &'a ty::ctxt<'tcx>
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

fn scan<R, F, G>(st: &mut PState, mut is_last: F, op: G) -> R where
    F: FnMut(char) -> bool,
    G: FnOnce(&[u8]) -> R,
{
    let start_pos = st.pos;
    debug!("scan: '{}' (start)", st.data[st.pos] as char);
    while !is_last(st.data[st.pos] as char) {
        st.pos += 1;
        debug!("scan: '{}'", st.data[st.pos] as char);
    }
    let end_pos = st.pos;
    st.pos += 1;
    return op(st.data[start_pos..end_pos]);
}

pub fn parse_ident(st: &mut PState, last: char) -> ast::Ident {
    ast::Ident::new(parse_name(st, last))
}

pub fn parse_name(st: &mut PState, last: char) -> ast::Name {
    fn is_last(b: char, c: char) -> bool { return c == b; }
    parse_name_(st, |a| is_last(last, a) )
}

fn parse_name_<F>(st: &mut PState, is_last: F) -> ast::Name where
    F: FnMut(char) -> bool,
{
    scan(st, is_last, |bytes| {
        token::intern(str::from_utf8(bytes).unwrap())
    })
}

pub fn parse_state_from_data<'a, 'tcx>(data: &'a [u8], crate_num: ast::CrateNum,
                                       pos: uint, tcx: &'a ty::ctxt<'tcx>)
                                       -> PState<'a, 'tcx> {
    PState {
        data: data,
        krate: crate_num,
        pos: pos,
        tcx: tcx
    }
}

fn data_log_string(data: &[u8], pos: uint) -> String {
    let mut buf = String::new();
    buf.push_str("<<");
    for i in range(pos, data.len()) {
        let c = data[i];
        if c > 0x20 && c <= 0x7F {
            buf.push(c as char);
        } else {
            buf.push('.');
        }
    }
    buf.push_str(">>");
    buf
}

pub fn parse_ty_closure_data<'tcx>(data: &[u8],
                                   crate_num: ast::CrateNum,
                                   pos: uint,
                                   tcx: &ty::ctxt<'tcx>,
                                   conv: conv_did)
                                   -> ty::ClosureTy<'tcx> {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_closure_ty(&mut st, conv)
}

pub fn parse_ty_data<'tcx>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                           tcx: &ty::ctxt<'tcx>, conv: conv_did) -> Ty<'tcx> {
    debug!("parse_ty_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_ty(&mut st, conv)
}

pub fn parse_region_data(data: &[u8], crate_num: ast::CrateNum, pos: uint, tcx: &ty::ctxt,
                         conv: conv_did) -> ty::Region {
    debug!("parse_region_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_region(&mut st, conv)
}

pub fn parse_bare_fn_ty_data<'tcx>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                                   tcx: &ty::ctxt<'tcx>, conv: conv_did)
                                   -> ty::BareFnTy<'tcx> {
    debug!("parse_bare_fn_ty_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_bare_fn_ty(&mut st, conv)
}

pub fn parse_trait_ref_data<'tcx>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                                  tcx: &ty::ctxt<'tcx>, conv: conv_did)
                                  -> ty::TraitRef<'tcx> {
    debug!("parse_trait_ref_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_trait_ref(&mut st, conv)
}

pub fn parse_substs_data<'tcx>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                               tcx: &ty::ctxt<'tcx>, conv: conv_did) -> subst::Substs<'tcx> {
    debug!("parse_substs_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_substs(&mut st, conv)
}

pub fn parse_bounds_data<'tcx>(data: &[u8], crate_num: ast::CrateNum,
                               pos: uint, tcx: &ty::ctxt<'tcx>, conv: conv_did)
                               -> ty::ParamBounds<'tcx> {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_bounds(&mut st, conv)
}

pub fn parse_existential_bounds_data(data: &[u8], crate_num: ast::CrateNum,
                                     pos: uint, tcx: &ty::ctxt, conv: conv_did)
                                     -> ty::ExistentialBounds {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_existential_bounds(&mut st, conv)
}

pub fn parse_builtin_bounds_data(data: &[u8], crate_num: ast::CrateNum,
                                 pos: uint, tcx: &ty::ctxt, conv: conv_did)
                                 -> ty::BuiltinBounds {
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_builtin_bounds(&mut st, conv)
}

fn parse_size(st: &mut PState) -> Option<uint> {
    assert_eq!(next(st), '/');

    if peek(st) == '|' {
        assert_eq!(next(st), '|');
        None
    } else {
        let n = parse_uint(st);
        assert_eq!(next(st), '|');
        Some(n)
    }
}

fn parse_trait_store(st: &mut PState, conv: conv_did) -> ty::TraitStore {
    match next(st) {
        '~' => ty::UniqTraitStore,
        '&' => ty::RegionTraitStore(parse_region(st, conv), parse_mutability(st)),
        c => {
            st.tcx.sess.bug(format!("parse_trait_store(): bad input '{}'",
                                    c)[])
        }
    }
}

fn parse_vec_per_param_space<'a, 'tcx, T, F>(st: &mut PState<'a, 'tcx>,
                                             mut f: F)
                                             -> VecPerParamSpace<T> where
    F: FnMut(&mut PState<'a, 'tcx>) -> T,
{
    let mut r = VecPerParamSpace::empty();
    for &space in subst::ParamSpace::all().iter() {
        assert_eq!(next(st), '[');
        while peek(st) != ']' {
            r.push(space, f(st));
        }
        assert_eq!(next(st), ']');
    }
    r
}

fn parse_substs<'a, 'tcx>(st: &mut PState<'a, 'tcx>,
                          conv: conv_did) -> subst::Substs<'tcx> {
    let regions =
        parse_region_substs(st, |x,y| conv(x,y));

    let types =
        parse_vec_per_param_space(st, |st| parse_ty(st, |x,y| conv(x,y)));

    return subst::Substs { types: types,
                           regions: regions };
}

fn parse_region_substs(st: &mut PState, conv: conv_did) -> subst::RegionSubsts {
    match next(st) {
        'e' => subst::ErasedRegions,
        'n' => {
            subst::NonerasedRegions(
                parse_vec_per_param_space(
                    st, |st| parse_region(st, |x,y| conv(x,y))))
        }
        _ => panic!("parse_bound_region: bad input")
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
            let ident = token::str_to_ident(parse_str(st, ']')[]);
            ty::BrNamed(def, ident.name)
        }
        'f' => {
            let id = parse_uint(st);
            assert_eq!(next(st), '|');
            ty::BrFresh(id)
        }
        'e' => ty::BrEnv,
        _ => panic!("parse_bound_region: bad input")
    }
}

fn parse_region(st: &mut PState, conv: conv_did) -> ty::Region {
    match next(st) {
      'b' => {
        assert_eq!(next(st), '[');
        let id = ty::DebruijnIndex::new(parse_uint(st));
        assert_eq!(next(st), '|');
        let br = parse_bound_region(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        ty::ReLateBound(id, br)
      }
      'B' => {
        assert_eq!(next(st), '[');
        let node_id = parse_uint(st) as ast::NodeId;
        assert_eq!(next(st), '|');
        let space = parse_param_space(st);
        assert_eq!(next(st), '|');
        let index = parse_uint(st);
        assert_eq!(next(st), '|');
        let nm = token::str_to_ident(parse_str(st, ']')[]);
        ty::ReEarlyBound(node_id, space, index, nm.name)
      }
      'f' => {
        assert_eq!(next(st), '[');
        let scope = parse_scope(st);
        assert_eq!(next(st), '|');
        let br = parse_bound_region(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        ty::ReFree(ty::FreeRegion { scope: scope,
                                    bound_region: br})
      }
      's' => {
        let scope = parse_scope(st);
        assert_eq!(next(st), '|');
        ty::ReScope(scope)
      }
      't' => {
        ty::ReStatic
      }
      'e' => {
        ty::ReStatic
      }
      _ => panic!("parse_region: bad input")
    }
}

fn parse_scope(st: &mut PState) -> region::CodeExtent {
    match next(st) {
        'M' => {
            let node_id = parse_uint(st) as ast::NodeId;
            region::CodeExtent::Misc(node_id)
        }
        _ => panic!("parse_scope: bad input")
    }
}

fn parse_opt<'a, 'tcx, T, F>(st: &mut PState<'a, 'tcx>, f: F) -> Option<T> where
    F: FnOnce(&mut PState<'a, 'tcx>) -> T,
{
    match next(st) {
      'n' => None,
      's' => Some(f(st)),
      _ => panic!("parse_opt: bad input")
    }
}

fn parse_str(st: &mut PState, term: char) -> String {
    let mut result = String::new();
    while peek(st) != term {
        unsafe {
            result.as_mut_vec().push_all(&[next_byte(st)])
        }
    }
    next(st);
    result
}

fn parse_trait_ref<'a, 'tcx>(st: &mut PState<'a, 'tcx>, conv: conv_did)
                             -> ty::TraitRef<'tcx> {
    let def = parse_def(st, NominalType, |x,y| conv(x,y));
    let substs = parse_substs(st, |x,y| conv(x,y));
    ty::TraitRef {def_id: def, substs: substs}
}

fn parse_ty<'a, 'tcx>(st: &mut PState<'a, 'tcx>, conv: conv_did) -> Ty<'tcx> {
    match next(st) {
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
          _ => panic!("parse_ty: bad numeric type")
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
        let trait_ref = ty::Binder(parse_trait_ref(st, |x,y| conv(x,y)));
        let bounds = parse_existential_bounds(st, |x,y| conv(x,y));
        assert_eq!(next(st), ']');
        return ty::mk_trait(st.tcx, trait_ref, bounds);
      }
      'p' => {
        let did = parse_def(st, TypeParameter, |x,y| conv(x,y));
        debug!("parsed ty_param: did={}", did);
        let index = parse_uint(st);
        assert_eq!(next(st), '|');
        let space = parse_param_space(st);
        assert_eq!(next(st), '|');
        return ty::mk_param(st.tcx, space, index, did);
      }
      '~' => return ty::mk_uniq(st.tcx, parse_ty(st, |x,y| conv(x,y))),
      '*' => return ty::mk_ptr(st.tcx, parse_mt(st, |x,y| conv(x,y))),
      '&' => {
        let r = parse_region(st, |x,y| conv(x,y));
        let mt = parse_mt(st, |x,y| conv(x,y));
        return ty::mk_rptr(st.tcx, r, mt);
      }
      'V' => {
        let t = parse_ty(st, |x,y| conv(x,y));
        let sz = parse_size(st);
        return ty::mk_vec(st.tcx, t, sz);
      }
      'v' => {
        return ty::mk_str(st.tcx);
      }
      'T' => {
        assert_eq!(next(st), '[');
        let mut params = Vec::new();
        while peek(st) != ']' { params.push(parse_ty(st, |x,y| conv(x,y))); }
        st.pos = st.pos + 1u;
        return ty::mk_tup(st.tcx, params);
      }
      'f' => {
        return ty::mk_closure(st.tcx, parse_closure_ty(st, |x,y| conv(x,y)));
      }
      'F' => {
          let def_id = parse_def(st, NominalType, |x,y| conv(x,y));
          return ty::mk_bare_fn(st.tcx, Some(def_id), parse_bare_fn_ty(st, |x,y| conv(x,y)));
      }
      'G' => {
          return ty::mk_bare_fn(st.tcx, None, parse_bare_fn_ty(st, |x,y| conv(x,y)));
      }
      '#' => {
        let pos = parse_hex(st);
        assert_eq!(next(st), ':');
        let len = parse_hex(st);
        assert_eq!(next(st), '#');
        let key = ty::creader_cache_key {cnum: st.krate,
                                         pos: pos,
                                         len: len };

        match st.tcx.rcache.borrow().get(&key).cloned() {
          Some(tt) => return tt,
          None => {}
        }
        let mut ps = PState {
            pos: pos,
            .. *st
        };
        let tt = parse_ty(&mut ps, |x,y| conv(x,y));
        st.tcx.rcache.borrow_mut().insert(key, tt);
        return tt;
      }
      '\"' => {
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
      'k' => {
          assert_eq!(next(st), '[');
          let did = parse_def(st, UnboxedClosureSource, |x,y| conv(x,y));
          let region = parse_region(st, |x,y| conv(x,y));
          let substs = parse_substs(st, |x,y| conv(x,y));
          assert_eq!(next(st), ']');
          return ty::mk_unboxed_closure(st.tcx, did, region, substs);
      }
      'e' => {
          return ty::mk_err();
      }
      c => { panic!("unexpected char in type string: {}", c);}
    }
}

fn parse_mutability(st: &mut PState) -> ast::Mutability {
    match peek(st) {
      'm' => { next(st); ast::MutMutable }
      _ => { ast::MutImmutable }
    }
}

fn parse_mt<'a, 'tcx>(st: &mut PState<'a, 'tcx>, conv: conv_did) -> ty::mt<'tcx> {
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

fn parse_param_space(st: &mut PState) -> subst::ParamSpace {
    subst::ParamSpace::from_uint(parse_uint(st))
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

fn parse_unsafety(c: char) -> ast::Unsafety {
    match c {
        'u' => ast::Unsafety::Unsafe,
        'n' => ast::Unsafety::Normal,
        _ => panic!("parse_unsafety: bad unsafety {}", c)
    }
}

fn parse_abi_set(st: &mut PState) -> abi::Abi {
    assert_eq!(next(st), '[');
    scan(st, |c| c == ']', |bytes| {
        let abi_str = str::from_utf8(bytes).unwrap();
        abi::lookup(abi_str[]).expect(abi_str)
    })
}

fn parse_onceness(c: char) -> ast::Onceness {
    match c {
        'o' => ast::Once,
        'm' => ast::Many,
        _ => panic!("parse_onceness: bad onceness")
    }
}

fn parse_closure_ty<'a, 'tcx>(st: &mut PState<'a, 'tcx>,
                              conv: conv_did) -> ty::ClosureTy<'tcx> {
    let unsafety = parse_unsafety(next(st));
    let onceness = parse_onceness(next(st));
    let store = parse_trait_store(st, |x,y| conv(x,y));
    let bounds = parse_existential_bounds(st, |x,y| conv(x,y));
    let sig = parse_sig(st, |x,y| conv(x,y));
    let abi = parse_abi_set(st);
    ty::ClosureTy {
        unsafety: unsafety,
        onceness: onceness,
        store: store,
        bounds: bounds,
        sig: sig,
        abi: abi,
    }
}

fn parse_bare_fn_ty<'a, 'tcx>(st: &mut PState<'a, 'tcx>,
                              conv: conv_did) -> ty::BareFnTy<'tcx> {
    let unsafety = parse_unsafety(next(st));
    let abi = parse_abi_set(st);
    let sig = parse_sig(st, |x,y| conv(x,y));
    ty::BareFnTy {
        unsafety: unsafety,
        abi: abi,
        sig: sig
    }
}

fn parse_sig<'a, 'tcx>(st: &mut PState<'a, 'tcx>, conv: conv_did) -> ty::PolyFnSig<'tcx> {
    assert_eq!(next(st), '[');
    let mut inputs = Vec::new();
    while peek(st) != ']' {
        inputs.push(parse_ty(st, |x,y| conv(x,y)));
    }
    st.pos += 1u; // eat the ']'
    let variadic = match next(st) {
        'V' => true,
        'N' => false,
        r => panic!(format!("bad variadic: {}", r)),
    };
    let output = match peek(st) {
        'z' => {
          st.pos += 1u;
          ty::FnDiverging
        }
        _ => ty::FnConverging(parse_ty(st, |x,y| conv(x,y)))
    };
    ty::Binder(ty::FnSig {inputs: inputs,
                        output: output,
                        variadic: variadic})
}

// Rust metadata parsing
pub fn parse_def_id(buf: &[u8]) -> ast::DefId {
    let mut colon_idx = 0u;
    let len = buf.len();
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1u; }
    if colon_idx == len {
        error!("didn't find ':' when parsing def id");
        panic!();
    }

    let crate_part = buf[0u..colon_idx];
    let def_part = buf[colon_idx + 1u..len];

    let crate_num = match str::from_utf8(crate_part).ok().and_then(|s| s.parse::<uint>()) {
       Some(cn) => cn as ast::CrateNum,
       None => panic!("internal error: parse_def_id: crate number expected, found {}",
                     crate_part)
    };
    let def_num = match str::from_utf8(def_part).ok().and_then(|s| s.parse::<uint>()) {
       Some(dn) => dn as ast::NodeId,
       None => panic!("internal error: parse_def_id: id expected, found {}",
                     def_part)
    };
    ast::DefId { krate: crate_num, node: def_num }
}

pub fn parse_predicate_data<'tcx>(data: &[u8],
                                  start: uint,
                                  crate_num: ast::CrateNum,
                                  tcx: &ty::ctxt<'tcx>,
                                  conv: conv_did)
                                  -> ty::Predicate<'tcx>
{
    let mut st = parse_state_from_data(data, crate_num, start, tcx);
    parse_predicate(&mut st, conv)
}

pub fn parse_predicate<'a,'tcx>(st: &mut PState<'a, 'tcx>,
                                conv: conv_did)
                                -> ty::Predicate<'tcx>
{
    match next(st) {
        't' => Rc::new(ty::Binder(parse_trait_ref(st, conv))).as_predicate(),
        'e' => ty::Binder(ty::EquatePredicate(parse_ty(st, |x,y| conv(x,y)),
                                              parse_ty(st, |x,y| conv(x,y)))).as_predicate(),
        'r' => ty::Binder(ty::OutlivesPredicate(parse_region(st, |x,y| conv(x,y)),
                                                parse_region(st, |x,y| conv(x,y)))).as_predicate(),
        'o' => ty::Binder(ty::OutlivesPredicate(parse_ty(st, |x,y| conv(x,y)),
                                                parse_region(st, |x,y| conv(x,y)))).as_predicate(),
        c => panic!("Encountered invalid character in metadata: {}", c)
    }
}

pub fn parse_type_param_def_data<'tcx>(data: &[u8], start: uint,
                                       crate_num: ast::CrateNum, tcx: &ty::ctxt<'tcx>,
                                       conv: conv_did) -> ty::TypeParameterDef<'tcx>
{
    let mut st = parse_state_from_data(data, crate_num, start, tcx);
    parse_type_param_def(&mut st, conv)
}

fn parse_type_param_def<'a, 'tcx>(st: &mut PState<'a, 'tcx>, conv: conv_did)
                                  -> ty::TypeParameterDef<'tcx> {
    let name = parse_name(st, ':');
    let def_id = parse_def(st, NominalType, |x,y| conv(x,y));
    let space = parse_param_space(st);
    assert_eq!(next(st), '|');
    let index = parse_uint(st);
    assert_eq!(next(st), '|');
    let associated_with = parse_opt(st, |st| {
        parse_def(st, NominalType, |x,y| conv(x,y))
    });
    assert_eq!(next(st), '|');
    let bounds = parse_bounds(st, |x,y| conv(x,y));
    let default = parse_opt(st, |st| parse_ty(st, |x,y| conv(x,y)));

    ty::TypeParameterDef {
        name: name,
        def_id: def_id,
        space: space,
        index: index,
        associated_with: associated_with,
        bounds: bounds,
        default: default
    }
}

fn parse_existential_bounds(st: &mut PState, conv: conv_did) -> ty::ExistentialBounds {
    let r = parse_region(st, |x,y| conv(x,y));
    let bb = parse_builtin_bounds(st, conv);
    return ty::ExistentialBounds { region_bound: r, builtin_bounds: bb };
}

fn parse_builtin_bounds(st: &mut PState, _conv: conv_did) -> ty::BuiltinBounds {
    let mut builtin_bounds = ty::empty_builtin_bounds();

    loop {
        match next(st) {
            'S' => {
                builtin_bounds.insert(ty::BoundSend);
            }
            'Z' => {
                builtin_bounds.insert(ty::BoundSized);
            }
            'P' => {
                builtin_bounds.insert(ty::BoundCopy);
            }
            'T' => {
                builtin_bounds.insert(ty::BoundSync);
            }
            '.' => {
                return builtin_bounds;
            }
            c => {
                panic!("parse_bounds: bad builtin bounds ('{}')", c)
            }
        }
    }
}

fn parse_bounds<'a, 'tcx>(st: &mut PState<'a, 'tcx>, conv: conv_did)
                          -> ty::ParamBounds<'tcx> {
    let builtin_bounds = parse_builtin_bounds(st, |x,y| conv(x,y));

    let mut param_bounds = ty::ParamBounds {
        region_bounds: Vec::new(),
        builtin_bounds: builtin_bounds,
        trait_bounds: Vec::new()
    };
    loop {
        match next(st) {
            'R' => {
                param_bounds.region_bounds.push(
                    parse_region(st, |x, y| conv (x, y)));
            }
            'I' => {
                param_bounds.trait_bounds.push(
                    Rc::new(ty::Binder(parse_trait_ref(st, |x,y| conv(x,y)))));
            }
            '.' => {
                return param_bounds;
            }
            c => {
                panic!("parse_bounds: bad bounds ('{}')", c)
            }
        }
    }
}
