#![allow(non_camel_case_types)]

use rustc_hir::LangItem;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Instance, TyCtxt};
use rustc_middle::{bug, mir, span_bug};
use rustc_session::cstore::{DllCallingConvention, DllImport, PeImportNameType};
use rustc_span::Span;
use rustc_target::spec::Target;

use crate::traits::*;

#[derive(Copy, Clone, Debug)]
pub enum IntPredicate {
    IntEQ,
    IntNE,
    IntUGT,
    IntUGE,
    IntULT,
    IntULE,
    IntSGT,
    IntSGE,
    IntSLT,
    IntSLE,
}

#[derive(Copy, Clone, Debug)]
pub enum RealPredicate {
    RealPredicateFalse,
    RealOEQ,
    RealOGT,
    RealOGE,
    RealOLT,
    RealOLE,
    RealONE,
    RealORD,
    RealUNO,
    RealUEQ,
    RealUGT,
    RealUGE,
    RealULT,
    RealULE,
    RealUNE,
    RealPredicateTrue,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AtomicRmwBinOp {
    AtomicXchg,
    AtomicAdd,
    AtomicSub,
    AtomicAnd,
    AtomicNand,
    AtomicOr,
    AtomicXor,
    AtomicMax,
    AtomicMin,
    AtomicUMax,
    AtomicUMin,
}

#[derive(Copy, Clone, Debug)]
pub enum SynchronizationScope {
    SingleThread,
    CrossThread,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TypeKind {
    Void,
    Half,
    Float,
    Double,
    X86_FP80,
    FP128,
    PPC_FP128,
    Label,
    Integer,
    Function,
    Struct,
    Array,
    Pointer,
    Vector,
    Metadata,
    Token,
    ScalableVector,
    BFloat,
    X86_AMX,
}

// FIXME(mw): Anything that is produced via DepGraph::with_task() must implement
//            the HashStable trait. Normally DepGraph::with_task() calls are
//            hidden behind queries, but CGU creation is a special case in two
//            ways: (1) it's not a query and (2) CGU are output nodes, so their
//            Fingerprints are not actually needed. It remains to be clarified
//            how exactly this case will be handled in the red/green system but
//            for now we content ourselves with providing a no-op HashStable
//            implementation for CGUs.
mod temp_stable_hash_impls {
    use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

    use crate::ModuleCodegen;

    impl<HCX, M> HashStable<HCX> for ModuleCodegen<M> {
        fn hash_stable(&self, _: &mut HCX, _: &mut StableHasher) {
            // do nothing
        }
    }
}

pub(crate) fn build_langcall<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &Bx,
    span: Option<Span>,
    li: LangItem,
) -> (Bx::FnAbiOfResult, Bx::Value, Instance<'tcx>) {
    let tcx = bx.tcx();
    let def_id = tcx.require_lang_item(li, span);
    let instance = ty::Instance::mono(tcx, def_id);
    (bx.fn_abi_of_instance(instance, ty::List::empty()), bx.get_fn_addr(instance), instance)
}

pub(crate) fn shift_mask_val<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    llty: Bx::Type,
    mask_llty: Bx::Type,
    invert: bool,
) -> Bx::Value {
    let kind = bx.type_kind(llty);
    match kind {
        TypeKind::Integer => {
            // i8/u8 can shift by at most 7, i16/u16 by at most 15, etc.
            let val = bx.int_width(llty) - 1;
            if invert {
                bx.const_int(mask_llty, !val as i64)
            } else {
                bx.const_uint(mask_llty, val)
            }
        }
        TypeKind::Vector => {
            let mask =
                shift_mask_val(bx, bx.element_type(llty), bx.element_type(mask_llty), invert);
            bx.vector_splat(bx.vector_length(mask_llty), mask)
        }
        _ => bug!("shift_mask_val: expected Integer or Vector, found {:?}", kind),
    }
}

pub fn asm_const_to_str<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    const_value: mir::ConstValue<'tcx>,
    ty_and_layout: TyAndLayout<'tcx>,
) -> String {
    let mir::ConstValue::Scalar(scalar) = const_value else {
        span_bug!(sp, "expected Scalar for promoted asm const, but got {:#?}", const_value)
    };
    let value = scalar.assert_scalar_int().to_bits(ty_and_layout.size);
    match ty_and_layout.ty.kind() {
        ty::Uint(_) => value.to_string(),
        ty::Int(int_ty) => match int_ty.normalize(tcx.sess.target.pointer_width) {
            ty::IntTy::I8 => (value as i8).to_string(),
            ty::IntTy::I16 => (value as i16).to_string(),
            ty::IntTy::I32 => (value as i32).to_string(),
            ty::IntTy::I64 => (value as i64).to_string(),
            ty::IntTy::I128 => (value as i128).to_string(),
            ty::IntTy::Isize => unreachable!(),
        },
        _ => span_bug!(sp, "asm const has bad type {}", ty_and_layout.ty),
    }
}

pub fn is_mingw_gnu_toolchain(target: &Target) -> bool {
    target.vendor == "pc" && target.os == "windows" && target.env == "gnu" && target.abi.is_empty()
}

pub fn i686_decorated_name(
    dll_import: &DllImport,
    mingw: bool,
    disable_name_mangling: bool,
    force_fully_decorated: bool,
) -> String {
    let name = dll_import.name.as_str();

    let (add_prefix, add_suffix) = match (force_fully_decorated, dll_import.import_name_type) {
        // No prefix is a bit weird, in that LLVM/ar_archive_writer won't emit it, so we will
        // ignore `force_fully_decorated` and always partially decorate it.
        (_, Some(PeImportNameType::NoPrefix)) => (false, true),
        (false, Some(PeImportNameType::Undecorated)) => (false, false),
        _ => (true, true),
    };

    // Worst case: +1 for disable name mangling, +1 for prefix, +4 for suffix (@@__).
    let mut decorated_name = String::with_capacity(name.len() + 6);

    if disable_name_mangling {
        // LLVM uses a binary 1 ('\x01') prefix to a name to indicate that mangling needs to be
        // disabled.
        decorated_name.push('\x01');
    }

    let prefix = if add_prefix && dll_import.is_fn {
        match dll_import.calling_convention {
            DllCallingConvention::C | DllCallingConvention::Vectorcall(_) => None,
            DllCallingConvention::Stdcall(_) => (!mingw
                || dll_import.import_name_type == Some(PeImportNameType::Decorated))
            .then_some('_'),
            DllCallingConvention::Fastcall(_) => Some('@'),
        }
    } else if !dll_import.is_fn && !mingw {
        // For static variables, prefix with '_' on MSVC.
        Some('_')
    } else {
        None
    };
    if let Some(prefix) = prefix {
        decorated_name.push(prefix);
    }

    decorated_name.push_str(name);

    if add_suffix && dll_import.is_fn {
        use std::fmt::Write;

        match dll_import.calling_convention {
            DllCallingConvention::C => {}
            DllCallingConvention::Stdcall(arg_list_size)
            | DllCallingConvention::Fastcall(arg_list_size) => {
                write!(&mut decorated_name, "@{arg_list_size}").unwrap();
            }
            DllCallingConvention::Vectorcall(arg_list_size) => {
                write!(&mut decorated_name, "@@{arg_list_size}").unwrap();
            }
        }
    }

    decorated_name
}
