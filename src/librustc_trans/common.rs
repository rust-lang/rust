// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types, non_snake_case)]

//! Code that is useful in various trans modules.

use llvm;
use llvm::{ValueRef, ContextRef, TypeKind};
use llvm::{True, False, Bool, OperandBundleDef};
use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;
use rustc::util::common::MemoizationMap;
use middle::lang_items::LangItem;
use base;
use builder::Builder;
use consts;
use declare;
use machine;
use monomorphize;
use type_::Type;
use value::Value;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::Layout;
use rustc::traits::{self, SelectionContext, Reveal};
use rustc::hir;

use libc::{c_uint, c_char};
use std::borrow::Cow;
use std::iter;

use syntax::ast;
use syntax::symbol::InternedString;
use syntax_pos::Span;

use rustc_i128::u128;

pub use context::{CrateContext, SharedCrateContext};

pub fn type_is_fat_ptr<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    if let Layout::FatPointer { .. } = *ccx.layout_of(ty) {
        true
    } else {
        false
    }
}

pub fn type_is_immediate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    let layout = ccx.layout_of(ty);
    match *layout {
        Layout::CEnum { .. } |
        Layout::Scalar { .. } |
        Layout::Vector { .. } => true,

        Layout::FatPointer { .. } => false,

        Layout::Array { .. } |
        Layout::Univariant { .. } |
        Layout::General { .. } |
        Layout::UntaggedUnion { .. } |
        Layout::RawNullablePointer { .. } |
        Layout::StructWrappedNullablePointer { .. } => {
            !layout.is_unsized() && layout.size(&ccx.tcx().data_layout).bytes() == 0
        }
    }
}

/// Returns Some([a, b]) if the type has a pair of fields with types a and b.
pub fn type_pair_fields<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>)
                                  -> Option<[Ty<'tcx>; 2]> {
    match ty.sty {
        ty::TyAdt(adt, substs) => {
            assert_eq!(adt.variants.len(), 1);
            let fields = &adt.variants[0].fields;
            if fields.len() != 2 {
                return None;
            }
            Some([monomorphize::field_ty(ccx.tcx(), substs, &fields[0]),
                  monomorphize::field_ty(ccx.tcx(), substs, &fields[1])])
        }
        ty::TyClosure(def_id, substs) => {
            let mut tys = substs.upvar_tys(def_id, ccx.tcx());
            tys.next().and_then(|first_ty| tys.next().and_then(|second_ty| {
                if tys.next().is_some() {
                    None
                } else {
                    Some([first_ty, second_ty])
                }
            }))
        }
        ty::TyTuple(tys) => {
            if tys.len() != 2 {
                return None;
            }
            Some([tys[0], tys[1]])
        }
        _ => None
    }
}

/// Returns true if the type is represented as a pair of immediates.
pub fn type_is_imm_pair<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>)
                                  -> bool {
    match *ccx.layout_of(ty) {
        Layout::FatPointer { .. } => true,
        Layout::Univariant { ref variant, .. } => {
            // There must be only 2 fields.
            if variant.offsets.len() != 2 {
                return false;
            }

            match type_pair_fields(ccx, ty) {
                Some([a, b]) => {
                    type_is_immediate(ccx, a) && type_is_immediate(ccx, b)
                }
                None => false
            }
        }
        _ => false
    }
}

/// Identify types which have size zero at runtime.
pub fn type_is_zero_size<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    use machine::llsize_of_alloc;
    use type_of::sizing_type_of;
    let llty = sizing_type_of(ccx, ty);
    llsize_of_alloc(ccx, llty) == 0
}

/*
* A note on nomenclature of linking: "extern", "foreign", and "upcall".
*
* An "extern" is an LLVM symbol we wind up emitting an undefined external
* reference to. This means "we don't have the thing in this compilation unit,
* please make sure you link it in at runtime". This could be a reference to
* C code found in a C library, or rust code found in a rust crate.
*
* Most "externs" are implicitly declared (automatically) as a result of a
* user declaring an extern _module_ dependency; this causes the rust driver
* to locate an extern crate, scan its compilation metadata, and emit extern
* declarations for any symbols used by the declaring crate.
*
* A "foreign" is an extern that references C (or other non-rust ABI) code.
* There is no metadata to scan for extern references so in these cases either
* a header-digester like bindgen, or manual function prototypes, have to
* serve as declarators. So these are usually given explicitly as prototype
* declarations, in rust code, with ABI attributes on them noting which ABI to
* link via.
*
* An "upcall" is a foreign call generated by the compiler (not corresponding
* to any user-written call in the code) into the runtime library, to perform
* some helper task such as bringing a task to life, allocating memory, etc.
*
*/

/// A structure representing an active landing pad for the duration of a basic
/// block.
///
/// Each `Block` may contain an instance of this, indicating whether the block
/// is part of a landing pad or not. This is used to make decision about whether
/// to emit `invoke` instructions (e.g. in a landing pad we don't continue to
/// use `invoke`) and also about various function call metadata.
///
/// For GNU exceptions (`landingpad` + `resume` instructions) this structure is
/// just a bunch of `None` instances (not too interesting), but for MSVC
/// exceptions (`cleanuppad` + `cleanupret` instructions) this contains data.
/// When inside of a landing pad, each function call in LLVM IR needs to be
/// annotated with which landing pad it's a part of. This is accomplished via
/// the `OperandBundleDef` value created for MSVC landing pads.
pub struct Funclet {
    cleanuppad: ValueRef,
    operand: OperandBundleDef,
}

impl Funclet {
    pub fn new(cleanuppad: ValueRef) -> Funclet {
        Funclet {
            cleanuppad: cleanuppad,
            operand: OperandBundleDef::new("funclet", &[cleanuppad]),
        }
    }

    pub fn cleanuppad(&self) -> ValueRef {
        self.cleanuppad
    }

    pub fn bundle(&self) -> &OperandBundleDef {
        &self.operand
    }
}

impl Clone for Funclet {
    fn clone(&self) -> Funclet {
        Funclet {
            cleanuppad: self.cleanuppad,
            operand: OperandBundleDef::new("funclet", &[self.cleanuppad]),
        }
    }
}

pub fn val_ty(v: ValueRef) -> Type {
    unsafe {
        Type::from_ref(llvm::LLVMTypeOf(v))
    }
}

// LLVM constant constructors.
pub fn C_null(t: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstNull(t.to_ref())
    }
}

pub fn C_undef(t: Type) -> ValueRef {
    unsafe {
        llvm::LLVMGetUndef(t.to_ref())
    }
}

pub fn C_integral(t: Type, u: u64, sign_extend: bool) -> ValueRef {
    unsafe {
        llvm::LLVMConstInt(t.to_ref(), u, sign_extend as Bool)
    }
}

pub fn C_big_integral(t: Type, u: u128, sign_extend: bool) -> ValueRef {
    if ::std::mem::size_of::<u128>() == 16 {
        unsafe {
            llvm::LLVMConstIntOfArbitraryPrecision(t.to_ref(), 2, &u as *const u128 as *const u64)
        }
    } else {
        // SNAP: remove after snapshot
        C_integral(t, u as u64, sign_extend)
    }
}

pub fn C_floating_f64(f: f64, t: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstReal(t.to_ref(), f)
    }
}

pub fn C_nil(ccx: &CrateContext) -> ValueRef {
    C_struct(ccx, &[], false)
}

pub fn C_bool(ccx: &CrateContext, val: bool) -> ValueRef {
    C_integral(Type::i1(ccx), val as u64, false)
}

pub fn C_i32(ccx: &CrateContext, i: i32) -> ValueRef {
    C_integral(Type::i32(ccx), i as u64, true)
}

pub fn C_u32(ccx: &CrateContext, i: u32) -> ValueRef {
    C_integral(Type::i32(ccx), i as u64, false)
}

pub fn C_u64(ccx: &CrateContext, i: u64) -> ValueRef {
    C_integral(Type::i64(ccx), i, false)
}

pub fn C_uint<I: AsU64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_u64();

    let bit_size = machine::llbitsize_of_real(ccx, ccx.int_type());

    if bit_size < 64 {
        // make sure it doesn't overflow
        assert!(v < (1<<bit_size));
    }

    C_integral(ccx.int_type(), v, false)
}

pub trait AsI64 { fn as_i64(self) -> i64; }
pub trait AsU64 { fn as_u64(self) -> u64; }

// FIXME: remove the intptr conversions, because they
// are host-architecture-dependent
impl AsI64 for i64 { fn as_i64(self) -> i64 { self as i64 }}
impl AsI64 for i32 { fn as_i64(self) -> i64 { self as i64 }}
impl AsI64 for isize { fn as_i64(self) -> i64 { self as i64 }}

impl AsU64 for u64  { fn as_u64(self) -> u64 { self as u64 }}
impl AsU64 for u32  { fn as_u64(self) -> u64 { self as u64 }}
impl AsU64 for usize { fn as_u64(self) -> u64 { self as u64 }}

pub fn C_u8(ccx: &CrateContext, i: u8) -> ValueRef {
    C_integral(Type::i8(ccx), i as u64, false)
}


// This is a 'c-like' raw string, which differs from
// our boxed-and-length-annotated strings.
pub fn C_cstr(cx: &CrateContext, s: InternedString, null_terminated: bool) -> ValueRef {
    unsafe {
        if let Some(&llval) = cx.const_cstr_cache().borrow().get(&s) {
            return llval;
        }

        let sc = llvm::LLVMConstStringInContext(cx.llcx(),
                                                s.as_ptr() as *const c_char,
                                                s.len() as c_uint,
                                                !null_terminated as Bool);
        let sym = cx.generate_local_symbol_name("str");
        let g = declare::define_global(cx, &sym[..], val_ty(sc)).unwrap_or_else(||{
            bug!("symbol `{}` is already defined", sym);
        });
        llvm::LLVMSetInitializer(g, sc);
        llvm::LLVMSetGlobalConstant(g, True);
        llvm::LLVMRustSetLinkage(g, llvm::Linkage::InternalLinkage);

        cx.const_cstr_cache().borrow_mut().insert(s, g);
        g
    }
}

// NB: Do not use `do_spill_noroot` to make this into a constant string, or
// you will be kicked off fast isel. See issue #4352 for an example of this.
pub fn C_str_slice(cx: &CrateContext, s: InternedString) -> ValueRef {
    let len = s.len();
    let cs = consts::ptrcast(C_cstr(cx, s, false), Type::i8p(cx));
    C_named_struct(cx.str_slice_type(), &[cs, C_uint(cx, len)])
}

pub fn C_struct(cx: &CrateContext, elts: &[ValueRef], packed: bool) -> ValueRef {
    C_struct_in_context(cx.llcx(), elts, packed)
}

pub fn C_struct_in_context(llcx: ContextRef, elts: &[ValueRef], packed: bool) -> ValueRef {
    unsafe {
        llvm::LLVMConstStructInContext(llcx,
                                       elts.as_ptr(), elts.len() as c_uint,
                                       packed as Bool)
    }
}

pub fn C_named_struct(t: Type, elts: &[ValueRef]) -> ValueRef {
    unsafe {
        llvm::LLVMConstNamedStruct(t.to_ref(), elts.as_ptr(), elts.len() as c_uint)
    }
}

pub fn C_array(ty: Type, elts: &[ValueRef]) -> ValueRef {
    unsafe {
        return llvm::LLVMConstArray(ty.to_ref(), elts.as_ptr(), elts.len() as c_uint);
    }
}

pub fn C_vector(elts: &[ValueRef]) -> ValueRef {
    unsafe {
        return llvm::LLVMConstVector(elts.as_ptr(), elts.len() as c_uint);
    }
}

pub fn C_bytes(cx: &CrateContext, bytes: &[u8]) -> ValueRef {
    C_bytes_in_context(cx.llcx(), bytes)
}

pub fn C_bytes_in_context(llcx: ContextRef, bytes: &[u8]) -> ValueRef {
    unsafe {
        let ptr = bytes.as_ptr() as *const c_char;
        return llvm::LLVMConstStringInContext(llcx, ptr, bytes.len() as c_uint, True);
    }
}

pub fn const_get_elt(v: ValueRef, us: &[c_uint])
              -> ValueRef {
    unsafe {
        let r = llvm::LLVMConstExtractValue(v, us.as_ptr(), us.len() as c_uint);

        debug!("const_get_elt(v={:?}, us={:?}, r={:?})",
               Value(v), us, Value(r));

        r
    }
}

pub fn const_to_uint(v: ValueRef) -> u64 {
    unsafe {
        llvm::LLVMConstIntGetZExtValue(v)
    }
}

fn is_const_integral(v: ValueRef) -> bool {
    unsafe {
        !llvm::LLVMIsAConstantInt(v).is_null()
    }
}

#[inline]
#[cfg(stage0)]
fn hi_lo_to_u128(lo: u64, _: u64) -> u128 {
    lo as u128
}

#[inline]
#[cfg(not(stage0))]
fn hi_lo_to_u128(lo: u64, hi: u64) -> u128 {
    ((hi as u128) << 64) | (lo as u128)
}

pub fn const_to_opt_u128(v: ValueRef, sign_ext: bool) -> Option<u128> {
    unsafe {
        if is_const_integral(v) {
            let (mut lo, mut hi) = (0u64, 0u64);
            let success = llvm::LLVMRustConstInt128Get(v, sign_ext,
                                                       &mut hi as *mut u64, &mut lo as *mut u64);
            if success {
                Some(hi_lo_to_u128(lo, hi))
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub fn is_undef(val: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsUndef(val) != False
    }
}

#[allow(dead_code)] // potentially useful
pub fn is_null(val: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsNull(val) != False
    }
}

/// Attempts to resolve an obligation. The result is a shallow vtable resolution -- meaning that we
/// do not (necessarily) resolve all nested obligations on the impl. Note that type check should
/// guarantee to us that all nested obligations *could be* resolved if we wanted to.
pub fn fulfill_obligation<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                    span: Span,
                                    trait_ref: ty::PolyTraitRef<'tcx>)
                                    -> traits::Vtable<'tcx, ()>
{
    let tcx = scx.tcx();

    // Remove any references to regions; this helps improve caching.
    let trait_ref = tcx.erase_regions(&trait_ref);

    scx.trait_cache().memoize(trait_ref, || {
        debug!("trans::fulfill_obligation(trait_ref={:?}, def_id={:?})",
               trait_ref, trait_ref.def_id());

        // Do the initial selection for the obligation. This yields the
        // shallow result we are looking for -- that is, what specific impl.
        tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
            let mut selcx = SelectionContext::new(&infcx);

            let obligation_cause = traits::ObligationCause::misc(span,
                                                             ast::DUMMY_NODE_ID);
            let obligation = traits::Obligation::new(obligation_cause,
                                                     trait_ref.to_poly_trait_predicate());

            let selection = match selcx.select(&obligation) {
                Ok(Some(selection)) => selection,
                Ok(None) => {
                    // Ambiguity can happen when monomorphizing during trans
                    // expands to some humongo type that never occurred
                    // statically -- this humongo type can then overflow,
                    // leading to an ambiguous result. So report this as an
                    // overflow bug, since I believe this is the only case
                    // where ambiguity can result.
                    debug!("Encountered ambiguity selecting `{:?}` during trans, \
                            presuming due to overflow",
                           trait_ref);
                    tcx.sess.span_fatal(span,
                        "reached the recursion limit during monomorphization \
                         (selection ambiguity)");
                }
                Err(e) => {
                    span_bug!(span, "Encountered error `{:?}` selecting `{:?}` during trans",
                              e, trait_ref)
                }
            };

            debug!("fulfill_obligation: selection={:?}", selection);

            // Currently, we use a fulfillment context to completely resolve
            // all nested obligations. This is because they can inform the
            // inference of the impl's type parameters.
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let vtable = selection.map(|predicate| {
                debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });
            let vtable = infcx.drain_fulfillment_cx_or_panic(span, &mut fulfill_cx, &vtable);

            info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
            vtable
        })
    })
}

pub fn langcall(tcx: TyCtxt,
                span: Option<Span>,
                msg: &str,
                li: LangItem)
                -> DefId {
    match tcx.lang_items.require(li) {
        Ok(id) => id,
        Err(s) => {
            let msg = format!("{} {}", msg, s);
            match span {
                Some(span) => tcx.sess.span_fatal(span, &msg[..]),
                None => tcx.sess.fatal(&msg[..]),
            }
        }
    }
}

// To avoid UB from LLVM, these two functions mask RHS with an
// appropriate mask unconditionally (i.e. the fallback behavior for
// all shifts). For 32- and 64-bit types, this matches the semantics
// of Java. (See related discussion on #1877 and #10183.)

pub fn build_unchecked_lshift<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    lhs: ValueRef,
    rhs: ValueRef
) -> ValueRef {
    let rhs = base::cast_shift_expr_rhs(bcx, hir::BinOp_::BiShl, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bcx, rhs);
    bcx.shl(lhs, rhs)
}

pub fn build_unchecked_rshift<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>, lhs_t: Ty<'tcx>, lhs: ValueRef, rhs: ValueRef
) -> ValueRef {
    let rhs = base::cast_shift_expr_rhs(bcx, hir::BinOp_::BiShr, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bcx, rhs);
    let is_signed = lhs_t.is_signed();
    if is_signed {
        bcx.ashr(lhs, rhs)
    } else {
        bcx.lshr(lhs, rhs)
    }
}

fn shift_mask_rhs<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, rhs: ValueRef) -> ValueRef {
    let rhs_llty = val_ty(rhs);
    bcx.and(rhs, shift_mask_val(bcx, rhs_llty, rhs_llty, false))
}

pub fn shift_mask_val<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    llty: Type,
    mask_llty: Type,
    invert: bool
) -> ValueRef {
    let kind = llty.kind();
    match kind {
        TypeKind::Integer => {
            // i8/u8 can shift by at most 7, i16/u16 by at most 15, etc.
            let val = llty.int_width() - 1;
            if invert {
                C_integral(mask_llty, !val, true)
            } else {
                C_integral(mask_llty, val, false)
            }
        },
        TypeKind::Vector => {
            let mask = shift_mask_val(bcx, llty.element_type(), mask_llty.element_type(), invert);
            bcx.vector_splat(mask_llty.vector_length(), mask)
        },
        _ => bug!("shift_mask_val: expected Integer or Vector, found {:?}", kind),
    }
}

pub fn ty_fn_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                          ty: Ty<'tcx>)
                          -> Cow<'tcx, ty::BareFnTy<'tcx>>
{
    match ty.sty {
        ty::TyFnDef(_, _, fty) => Cow::Borrowed(fty),
        // Shims currently have type TyFnPtr. Not sure this should remain.
        ty::TyFnPtr(fty) => Cow::Borrowed(fty),
        ty::TyClosure(def_id, substs) => {
            let tcx = ccx.tcx();
            let ty::ClosureTy { unsafety, abi, sig } = tcx.closure_type(def_id, substs);

            let env_region = ty::ReLateBound(ty::DebruijnIndex::new(1), ty::BrEnv);
            let env_ty = match tcx.closure_kind(def_id) {
                ty::ClosureKind::Fn => tcx.mk_imm_ref(tcx.mk_region(env_region), ty),
                ty::ClosureKind::FnMut => tcx.mk_mut_ref(tcx.mk_region(env_region), ty),
                ty::ClosureKind::FnOnce => ty,
            };

            let sig = sig.map_bound(|sig| tcx.mk_fn_sig(
                iter::once(env_ty).chain(sig.inputs().iter().cloned()),
                sig.output(),
                sig.variadic
            ));
            Cow::Owned(ty::BareFnTy { unsafety: unsafety, abi: abi, sig: sig })
        }
        _ => bug!("unexpected type {:?} to ty_fn_sig", ty)
    }
}

pub fn is_closure(tcx: TyCtxt, def_id: DefId) -> bool {
    tcx.def_key(def_id).disambiguated_data.data == DefPathData::ClosureExpr
}
