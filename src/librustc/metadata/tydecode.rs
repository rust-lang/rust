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
use middle::ty::{self, AsPredicate, Ty};

use std::rc::Rc;
use std::str;
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
#[derive(Copy, Debug)]
pub enum DefIdSource {
    // Identifies a struct, trait, enum, etc.
    NominalType,

    // Identifies a type alias (`type X = ...`).
    TypeWithId,

    // Identifies a type parameter (`fn foo<X>() { ... }`).
    TypeParameter,

    // Identifies a region parameter (`fn foo<'X>() { ... }`).
    RegionParameter,

    // Identifies a closure
    ClosureSource
}

// type conv_did = impl FnMut(DefIdSource, ast::DefId) -> ast::DefId;

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
    st.pos = st.pos + 1;
    return ch;
}

fn next_byte(st: &mut PState) -> u8 {
    let b = st.data[st.pos];
    st.pos = st.pos + 1;
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
    return op(&st.data[start_pos..end_pos]);
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
    for i in pos..data.len() {
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

pub fn parse_ty_closure_data<'tcx, F>(data: &[u8],
                                      crate_num: ast::CrateNum,
                                      pos: uint,
                                      tcx: &ty::ctxt<'tcx>,
                                      conv: F)
                                      -> ty::ClosureTy<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_closure_ty(&mut st, conv)
}

pub fn parse_ty_data<'tcx, F>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                              tcx: &ty::ctxt<'tcx>, conv: F) -> Ty<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    debug!("parse_ty_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_ty(&mut st, conv)
}

pub fn parse_region_data<F>(data: &[u8], crate_num: ast::CrateNum, pos: uint, tcx: &ty::ctxt,
                            conv: F) -> ty::Region where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    debug!("parse_region_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_region(&mut st, conv)
}

pub fn parse_bare_fn_ty_data<'tcx, F>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                                      tcx: &ty::ctxt<'tcx>, conv: F)
                                      -> ty::BareFnTy<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    debug!("parse_bare_fn_ty_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_bare_fn_ty(&mut st, conv)
}

pub fn parse_trait_ref_data<'tcx, F>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                                     tcx: &ty::ctxt<'tcx>, conv: F)
                                     -> Rc<ty::TraitRef<'tcx>> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    debug!("parse_trait_ref_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_trait_ref(&mut st, conv)
}

pub fn parse_substs_data<'tcx, F>(data: &[u8], crate_num: ast::CrateNum, pos: uint,
                                  tcx: &ty::ctxt<'tcx>, conv: F) -> subst::Substs<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    debug!("parse_substs_data {}", data_log_string(data, pos));
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_substs(&mut st, conv)
}

pub fn parse_bounds_data<'tcx, F>(data: &[u8], crate_num: ast::CrateNum,
                                  pos: uint, tcx: &ty::ctxt<'tcx>, conv: F)
                                  -> ty::ParamBounds<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_bounds(&mut st, conv)
}

pub fn parse_existential_bounds_data<'tcx, F>(data: &[u8], crate_num: ast::CrateNum,
                                              pos: uint, tcx: &ty::ctxt<'tcx>, conv: F)
                                              -> ty::ExistentialBounds<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let mut st = parse_state_from_data(data, crate_num, pos, tcx);
    parse_existential_bounds(&mut st, conv)
}

pub fn parse_builtin_bounds_data<F>(data: &[u8], crate_num: ast::CrateNum,
                                    pos: uint, tcx: &ty::ctxt, conv: F)
                                    -> ty::BuiltinBounds where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
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

fn parse_vec_per_param_space<'a, 'tcx, T, F>(st: &mut PState<'a, 'tcx>,
                                             mut f: F)
                                             -> VecPerParamSpace<T> where
    F: FnMut(&mut PState<'a, 'tcx>) -> T,
{
    let mut r = VecPerParamSpace::empty();
    for &space in &subst::ParamSpace::all() {
        assert_eq!(next(st), '[');
        while peek(st) != ']' {
            r.push(space, f(st));
        }
        assert_eq!(next(st), ']');
    }
    r
}

fn parse_substs<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>,
                             mut conv: F) -> subst::Substs<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_substs_(st, &mut conv)
}

fn parse_substs_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>,
                              conv: &mut F) -> subst::Substs<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let regions =
        parse_region_substs_(st, conv);

    let types =
        parse_vec_per_param_space(st, |st| parse_ty_(st, conv));

    subst::Substs { types: types,
                    regions: regions }
}

fn parse_region_substs_<F>(st: &mut PState, conv: &mut F) -> subst::RegionSubsts where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    match next(st) {
        'e' => subst::ErasedRegions,
        'n' => {
            subst::NonerasedRegions(
                parse_vec_per_param_space(
                    st, |st| parse_region_(st, conv)))
        }
        _ => panic!("parse_bound_region: bad input")
    }
}

fn parse_bound_region_<F>(st: &mut PState, conv: &mut F) -> ty::BoundRegion where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    match next(st) {
        'a' => {
            let id = parse_u32(st);
            assert_eq!(next(st), '|');
            ty::BrAnon(id)
        }
        '[' => {
            let def = parse_def_(st, RegionParameter, conv);
            let ident = token::str_to_ident(&parse_str(st, ']')[]);
            ty::BrNamed(def, ident.name)
        }
        'f' => {
            let id = parse_u32(st);
            assert_eq!(next(st), '|');
            ty::BrFresh(id)
        }
        'e' => ty::BrEnv,
        _ => panic!("parse_bound_region: bad input")
    }
}

fn parse_region<F>(st: &mut PState, mut conv: F) -> ty::Region where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_region_(st, &mut conv)
}

fn parse_region_<F>(st: &mut PState, conv: &mut F) -> ty::Region where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    match next(st) {
      'b' => {
        assert_eq!(next(st), '[');
        let id = ty::DebruijnIndex::new(parse_u32(st));
        assert_eq!(next(st), '|');
        let br = parse_bound_region_(st, conv);
        assert_eq!(next(st), ']');
        ty::ReLateBound(id, br)
      }
      'B' => {
        assert_eq!(next(st), '[');
        let node_id = parse_uint(st) as ast::NodeId;
        assert_eq!(next(st), '|');
        let space = parse_param_space(st);
        assert_eq!(next(st), '|');
        let index = parse_u32(st);
        assert_eq!(next(st), '|');
        let nm = token::str_to_ident(&parse_str(st, ']')[]);
        ty::ReEarlyBound(node_id, space, index, nm.name)
      }
      'f' => {
        assert_eq!(next(st), '[');
        let scope = parse_destruction_scope_data(st);
        assert_eq!(next(st), '|');
        let br = parse_bound_region_(st, conv);
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
        'D' => {
            let node_id = parse_uint(st) as ast::NodeId;
            region::CodeExtent::DestructionScope(node_id)
        }
        'B' => {
            let node_id = parse_uint(st) as ast::NodeId;
            let first_stmt_index = parse_uint(st);
            let block_remainder = region::BlockRemainder {
                block: node_id, first_statement_index: first_stmt_index,
            };
            region::CodeExtent::Remainder(block_remainder)
        }
        _ => panic!("parse_scope: bad input")
    }
}

fn parse_destruction_scope_data(st: &mut PState) -> region::DestructionScopeData {
    let node_id = parse_uint(st) as ast::NodeId;
    region::DestructionScopeData::new(node_id)
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

fn parse_trait_ref<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, mut conv: F)
                                -> Rc<ty::TraitRef<'tcx>> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_trait_ref_(st, &mut conv)
}

fn parse_trait_ref_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, conv: &mut F)
                              -> Rc<ty::TraitRef<'tcx>> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let def = parse_def_(st, NominalType, conv);
    let substs = st.tcx.mk_substs(parse_substs_(st, conv));
    Rc::new(ty::TraitRef {def_id: def, substs: substs})
}

fn parse_ty<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, mut conv: F) -> Ty<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_ty_(st, &mut conv)
}

fn parse_ty_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, conv: &mut F) -> Ty<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let tcx = st.tcx;
    match next(st) {
      'b' => return tcx.types.bool,
      'i' => { /* eat the s of is */ next(st); return tcx.types.int },
      'u' => { /* eat the s of us */ next(st); return tcx.types.uint },
      'M' => {
        match next(st) {
          'b' => return tcx.types.u8,
          'w' => return tcx.types.u16,
          'l' => return tcx.types.u32,
          'd' => return tcx.types.u64,
          'B' => return tcx.types.i8,
          'W' => return tcx.types.i16,
          'L' => return tcx.types.i32,
          'D' => return tcx.types.i64,
          'f' => return tcx.types.f32,
          'F' => return tcx.types.f64,
          _ => panic!("parse_ty: bad numeric type")
        }
      }
      'c' => return tcx.types.char,
      't' => {
        assert_eq!(next(st), '[');
        let def = parse_def_(st, NominalType, conv);
        let substs = parse_substs_(st, conv);
        assert_eq!(next(st), ']');
        return ty::mk_enum(tcx, def, st.tcx.mk_substs(substs));
      }
      'x' => {
        assert_eq!(next(st), '[');
        let trait_ref = ty::Binder(parse_trait_ref_(st, conv));
        let bounds = parse_existential_bounds_(st, conv);
        assert_eq!(next(st), ']');
        return ty::mk_trait(tcx, trait_ref, bounds);
      }
      'p' => {
        assert_eq!(next(st), '[');
        let index = parse_u32(st);
        assert_eq!(next(st), '|');
        let space = parse_param_space(st);
        assert_eq!(next(st), '|');
        let name = token::intern(&parse_str(st, ']')[]);
        return ty::mk_param(tcx, space, index, name);
      }
      '~' => return ty::mk_uniq(tcx, parse_ty_(st, conv)),
      '*' => return ty::mk_ptr(tcx, parse_mt_(st, conv)),
      '&' => {
        let r = parse_region_(st, conv);
        let mt = parse_mt_(st, conv);
        return ty::mk_rptr(tcx, tcx.mk_region(r), mt);
      }
      'V' => {
        let t = parse_ty_(st, conv);
        let sz = parse_size(st);
        return ty::mk_vec(tcx, t, sz);
      }
      'v' => {
        return ty::mk_str(tcx);
      }
      'T' => {
        assert_eq!(next(st), '[');
        let mut params = Vec::new();
        while peek(st) != ']' { params.push(parse_ty_(st, conv)); }
        st.pos = st.pos + 1;
        return ty::mk_tup(tcx, params);
      }
      'F' => {
          let def_id = parse_def_(st, NominalType, conv);
          return ty::mk_bare_fn(tcx, Some(def_id),
                                tcx.mk_bare_fn(parse_bare_fn_ty_(st, conv)));
      }
      'G' => {
          return ty::mk_bare_fn(tcx, None,
                                tcx.mk_bare_fn(parse_bare_fn_ty_(st, conv)));
      }
      '#' => {
        let pos = parse_hex(st);
        assert_eq!(next(st), ':');
        let len = parse_hex(st);
        assert_eq!(next(st), '#');
        let key = ty::creader_cache_key {cnum: st.krate,
                                         pos: pos,
                                         len: len };

        match tcx.rcache.borrow().get(&key).cloned() {
          Some(tt) => return tt,
          None => {}
        }
        let mut ps = PState {
            pos: pos,
            .. *st
        };
        let tt = parse_ty_(&mut ps, conv);
        tcx.rcache.borrow_mut().insert(key, tt);
        return tt;
      }
      '\"' => {
        let _ = parse_def_(st, TypeWithId, conv);
        let inner = parse_ty_(st, conv);
        inner
      }
      'a' => {
          assert_eq!(next(st), '[');
          let did = parse_def_(st, NominalType, conv);
          let substs = parse_substs_(st, conv);
          assert_eq!(next(st), ']');
          return ty::mk_struct(st.tcx, did, st.tcx.mk_substs(substs));
      }
      'k' => {
          assert_eq!(next(st), '[');
          let did = parse_def_(st, ClosureSource, conv);
          let region = parse_region_(st, conv);
          let substs = parse_substs_(st, conv);
          assert_eq!(next(st), ']');
          return ty::mk_closure(st.tcx, did,
                  st.tcx.mk_region(region), st.tcx.mk_substs(substs));
      }
      'P' => {
          assert_eq!(next(st), '[');
          let trait_ref = parse_trait_ref_(st, conv);
          let name = token::intern(&parse_str(st, ']'));
          return ty::mk_projection(tcx, trait_ref, name);
      }
      'e' => {
          return tcx.types.err;
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

fn parse_mt_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, conv: &mut F) -> ty::mt<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let m = parse_mutability(st);
    ty::mt { ty: parse_ty_(st, conv), mutbl: m }
}

fn parse_def_<F>(st: &mut PState, source: DefIdSource, conv: &mut F) -> ast::DefId where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    return (*conv)(source, scan(st, |c| { c == '|' }, parse_def_id));
}

fn parse_uint(st: &mut PState) -> uint {
    let mut n = 0;
    loop {
        let cur = peek(st);
        if cur < '0' || cur > '9' { return n; }
        st.pos = st.pos + 1;
        n *= 10;
        n += (cur as uint) - ('0' as uint);
    };
}

fn parse_u32(st: &mut PState) -> u32 {
    let n = parse_uint(st);
    let m = n as u32;
    assert_eq!(m as uint, n);
    m
}

fn parse_param_space(st: &mut PState) -> subst::ParamSpace {
    subst::ParamSpace::from_uint(parse_uint(st))
}

fn parse_hex(st: &mut PState) -> uint {
    let mut n = 0;
    loop {
        let cur = peek(st);
        if (cur < '0' || cur > '9') && (cur < 'a' || cur > 'f') { return n; }
        st.pos = st.pos + 1;
        n *= 16;
        if '0' <= cur && cur <= '9' {
            n += (cur as uint) - ('0' as uint);
        } else { n += 10 + (cur as uint) - ('a' as uint); }
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
        abi::lookup(&abi_str[..]).expect(abi_str)
    })
}

fn parse_closure_ty<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>,
                                 mut conv: F) -> ty::ClosureTy<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_closure_ty_(st, &mut conv)
}

fn parse_closure_ty_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>,
                                 conv: &mut F) -> ty::ClosureTy<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let unsafety = parse_unsafety(next(st));
    let sig = parse_sig_(st, conv);
    let abi = parse_abi_set(st);
    ty::ClosureTy {
        unsafety: unsafety,
        sig: sig,
        abi: abi,
    }
}

fn parse_bare_fn_ty<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>,
                                 mut conv: F) -> ty::BareFnTy<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_bare_fn_ty_(st, &mut conv)
}

fn parse_bare_fn_ty_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>,
                                 conv: &mut F) -> ty::BareFnTy<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let unsafety = parse_unsafety(next(st));
    let abi = parse_abi_set(st);
    let sig = parse_sig_(st, conv);
    ty::BareFnTy {
        unsafety: unsafety,
        abi: abi,
        sig: sig
    }
}

fn parse_sig_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, conv: &mut F) -> ty::PolyFnSig<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    assert_eq!(next(st), '[');
    let mut inputs = Vec::new();
    while peek(st) != ']' {
        inputs.push(parse_ty_(st, conv));
    }
    st.pos += 1; // eat the ']'
    let variadic = match next(st) {
        'V' => true,
        'N' => false,
        r => panic!(format!("bad variadic: {}", r)),
    };
    let output = match peek(st) {
        'z' => {
          st.pos += 1;
          ty::FnDiverging
        }
        _ => ty::FnConverging(parse_ty_(st, conv))
    };
    ty::Binder(ty::FnSig {inputs: inputs,
                        output: output,
                        variadic: variadic})
}

// Rust metadata parsing
pub fn parse_def_id(buf: &[u8]) -> ast::DefId {
    let mut colon_idx = 0;
    let len = buf.len();
    while colon_idx < len && buf[colon_idx] != ':' as u8 { colon_idx += 1; }
    if colon_idx == len {
        error!("didn't find ':' when parsing def id");
        panic!();
    }

    let crate_part = &buf[0..colon_idx];
    let def_part = &buf[colon_idx + 1..len];

    let crate_num = match str::from_utf8(crate_part).ok().and_then(|s| {
        s.parse::<uint>().ok()
    }) {
       Some(cn) => cn as ast::CrateNum,
       None => panic!("internal error: parse_def_id: crate number expected, found {:?}",
                     crate_part)
    };
    let def_num = match str::from_utf8(def_part).ok().and_then(|s| {
        s.parse::<uint>().ok()
    }) {
       Some(dn) => dn as ast::NodeId,
       None => panic!("internal error: parse_def_id: id expected, found {:?}",
                     def_part)
    };
    ast::DefId { krate: crate_num, node: def_num }
}

pub fn parse_predicate_data<'tcx, F>(data: &[u8],
                                     start: uint,
                                     crate_num: ast::CrateNum,
                                     tcx: &ty::ctxt<'tcx>,
                                     conv: F)
                                     -> ty::Predicate<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let mut st = parse_state_from_data(data, crate_num, start, tcx);
    parse_predicate(&mut st, conv)
}

pub fn parse_predicate<'a,'tcx, F>(st: &mut PState<'a, 'tcx>,
                                   mut conv: F)
                                   -> ty::Predicate<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_predicate_(st, &mut conv)
}

fn parse_predicate_<'a,'tcx, F>(st: &mut PState<'a, 'tcx>,
                                conv: &mut F)
                                -> ty::Predicate<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    match next(st) {
        't' => ty::Binder(parse_trait_ref_(st, conv)).as_predicate(),
        'e' => ty::Binder(ty::EquatePredicate(parse_ty_(st, conv),
                                              parse_ty_(st, conv))).as_predicate(),
        'r' => ty::Binder(ty::OutlivesPredicate(parse_region_(st, conv),
                                                parse_region_(st, conv))).as_predicate(),
        'o' => ty::Binder(ty::OutlivesPredicate(parse_ty_(st, conv),
                                                parse_region_(st, conv))).as_predicate(),
        'p' => ty::Binder(parse_projection_predicate_(st, conv)).as_predicate(),
        c => panic!("Encountered invalid character in metadata: {}", c)
    }
}

fn parse_projection_predicate_<'a,'tcx, F>(
    st: &mut PState<'a, 'tcx>,
    conv: &mut F,
) -> ty::ProjectionPredicate<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    ty::ProjectionPredicate {
        projection_ty: ty::ProjectionTy {
            trait_ref: parse_trait_ref_(st, conv),
            item_name: token::str_to_ident(&parse_str(st, '|')).name,
        },
        ty: parse_ty_(st, conv),
    }
}

pub fn parse_type_param_def_data<'tcx, F>(data: &[u8], start: uint,
                                          crate_num: ast::CrateNum, tcx: &ty::ctxt<'tcx>,
                                          conv: F) -> ty::TypeParameterDef<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let mut st = parse_state_from_data(data, crate_num, start, tcx);
    parse_type_param_def(&mut st, conv)
}

fn parse_type_param_def<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, mut conv: F)
                                     -> ty::TypeParameterDef<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_type_param_def_(st, &mut conv)
}

fn parse_type_param_def_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, conv: &mut F)
                                      -> ty::TypeParameterDef<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let name = parse_name(st, ':');
    let def_id = parse_def_(st, NominalType, conv);
    let space = parse_param_space(st);
    assert_eq!(next(st), '|');
    let index = parse_u32(st);
    assert_eq!(next(st), '|');
    let bounds = parse_bounds_(st, conv);
    let default = parse_opt(st, |st| parse_ty_(st, conv));
    let object_lifetime_default = parse_object_lifetime_default(st, conv);

    ty::TypeParameterDef {
        name: name,
        def_id: def_id,
        space: space,
        index: index,
        bounds: bounds,
        default: default,
        object_lifetime_default: object_lifetime_default,
    }
}

fn parse_object_lifetime_default<'a,'tcx, F>(st: &mut PState<'a,'tcx>,
                                             conv: &mut F)
                                             -> Option<ty::ObjectLifetimeDefault>
    where F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    match next(st) {
        'n' => None,
        'a' => Some(ty::ObjectLifetimeDefault::Ambiguous),
        's' => {
            let region = parse_region_(st, conv);
            Some(ty::ObjectLifetimeDefault::Specific(region))
        }
        _ => panic!("parse_object_lifetime_default: bad input")
    }
}

fn parse_existential_bounds<'a,'tcx, F>(st: &mut PState<'a,'tcx>,
                                        mut conv: F)
                                        -> ty::ExistentialBounds<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_existential_bounds_(st, &mut conv)
}

fn parse_existential_bounds_<'a,'tcx, F>(st: &mut PState<'a,'tcx>,
                                        conv: &mut F)
                                        -> ty::ExistentialBounds<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let ty::ParamBounds { trait_bounds, mut region_bounds, builtin_bounds, projection_bounds } =
         parse_bounds_(st, conv);
    assert_eq!(region_bounds.len(), 1);
    assert_eq!(trait_bounds.len(), 0);
    let region_bound = region_bounds.pop().unwrap();
    return ty::ExistentialBounds { region_bound: region_bound,
                                   builtin_bounds: builtin_bounds,
                                   projection_bounds: projection_bounds };
}

fn parse_builtin_bounds<F>(st: &mut PState, mut _conv: F) -> ty::BuiltinBounds where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_builtin_bounds_(st, &mut _conv)
}

fn parse_builtin_bounds_<F>(st: &mut PState, _conv: &mut F) -> ty::BuiltinBounds where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
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

fn parse_bounds<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, mut conv: F)
                             -> ty::ParamBounds<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    parse_bounds_(st, &mut conv)
}

fn parse_bounds_<'a, 'tcx, F>(st: &mut PState<'a, 'tcx>, conv: &mut F)
                              -> ty::ParamBounds<'tcx> where
    F: FnMut(DefIdSource, ast::DefId) -> ast::DefId,
{
    let builtin_bounds = parse_builtin_bounds_(st, conv);

    let mut param_bounds = ty::ParamBounds {
        region_bounds: Vec::new(),
        builtin_bounds: builtin_bounds,
        trait_bounds: Vec::new(),
        projection_bounds: Vec::new(),
    };
    loop {
        match next(st) {
            'R' => {
                param_bounds.region_bounds.push(
                    parse_region_(st, conv));
            }
            'I' => {
                param_bounds.trait_bounds.push(
                    ty::Binder(parse_trait_ref_(st, conv)));
            }
            'P' => {
                param_bounds.projection_bounds.push(
                    ty::Binder(parse_projection_predicate_(st, conv)));
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
