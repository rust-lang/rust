//! Interface of a Rust codegen backend
//!
//! This crate defines all the traits that have to be implemented by a codegen backend in order to
//! use the backend-agnostic codegen code in `rustc_codegen_ssa`.
//!
//! The interface is designed around two backend-specific data structures, the codegen context and
//! the builder. The codegen context is supposed to be read-only after its creation and during the
//! actual codegen, while the builder stores the information about the function during codegen and
//! is used to produce the instructions of the backend IR.
//!
//! The traits contain associated types that are backend-specific, such as the backend's value or
//! basic blocks.

mod abi;
mod asm;
mod backend;
mod builder;
mod consts;
mod coverageinfo;
mod debuginfo;
mod declare;
mod intrinsic;
mod misc;
mod statics;
mod type_;
mod write;

use std::fmt;

use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, TyAndLayout};
use rustc_target::callconv::FnAbi;

pub use self::abi::AbiBuilderMethods;
pub use self::asm::{
    AsmBuilderMethods, AsmCodegenMethods, GlobalAsmOperandRef, InlineAsmOperandRef,
};
pub use self::backend::{BackendTypes, CodegenBackend, ExtraBackendMethods};
pub use self::builder::{BuilderMethods, OverflowOp};
pub use self::consts::ConstCodegenMethods;
pub use self::coverageinfo::CoverageInfoBuilderMethods;
pub use self::debuginfo::{DebugInfoBuilderMethods, DebugInfoCodegenMethods};
pub use self::declare::PreDefineCodegenMethods;
pub use self::intrinsic::IntrinsicCallBuilderMethods;
pub use self::misc::MiscCodegenMethods;
pub use self::statics::{StaticBuilderMethods, StaticCodegenMethods};
pub use self::type_::{
    ArgAbiBuilderMethods, BaseTypeCodegenMethods, DerivedTypeCodegenMethods,
    LayoutTypeCodegenMethods, TypeCodegenMethods, TypeMembershipCodegenMethods,
};
pub use self::write::{ModuleBufferMethods, ThinBufferMethods, WriteBackendMethods};

pub trait CodegenObject = Copy + fmt::Debug;

pub trait CodegenMethods<'tcx> = LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>>
    + FnAbiOf<'tcx, FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>>
    + TypeCodegenMethods<'tcx>
    + ConstCodegenMethods
    + StaticCodegenMethods
    + DebugInfoCodegenMethods<'tcx>
    + AsmCodegenMethods<'tcx>
    + PreDefineCodegenMethods<'tcx>;
