//! Backend-independent descriptions of type debuginfo.
//!
//! Every codegen unit has its own LLVM context, so debuginfo nodes cannot be shared between them.
//! The rustc-side work needed to build those nodes (names, layouts, unique IDs, ...) is the same
//! for every codegen unit, though, so it is computed once per type and stored here, and each
//! codegen unit only translates the stored description into backend nodes.

use std::sync::Arc;

use rustc_abi::{Align, Size};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_hir::def_id::DefId;
use rustc_span::Symbol;

use crate::ty::Ty;

#[derive(Debug, Clone)]
pub enum DebuginfoFieldName {
    Symbol(Symbol),
    Static(&'static str),
    Owned(String),
}

impl DebuginfoFieldName {
    pub fn as_str(&self) -> &str {
        match self {
            DebuginfoFieldName::Symbol(sym) => sym.as_str(),
            DebuginfoFieldName::Static(s) => s,
            DebuginfoFieldName::Owned(s) => s,
        }
    }
}

/// The data needed to create the (initially empty) composite node of a type.
#[derive(Debug)]
pub struct DebuginfoStubDesc {
    pub name: String,
    pub unique_id: String,
    pub size: Size,
    pub align: Align,
    /// The item whose namespace is the containing scope, if any.
    pub scope_item: Option<DefId>,
    /// Backend-specific flag bits.
    pub flags: u32,
}

#[derive(Debug)]
pub struct DebuginfoMemberDesc<'tcx> {
    pub name: DebuginfoFieldName,
    pub size: Size,
    pub align: Align,
    pub offset: Size,
    pub flags: u32,
    pub ty: Ty<'tcx>,
}

#[derive(Debug)]
pub struct DebuginfoGenericParamDesc<'tcx> {
    pub name: Symbol,
    pub ty: Ty<'tcx>,
}

#[derive(Debug)]
pub struct DebuginfoVariantDesc<'tcx> {
    pub name: Symbol,
    pub struct_unique_id: String,
    pub members: Vec<DebuginfoMemberDesc<'tcx>>,
    pub discr: Option<u128>,
}

#[derive(Debug)]
pub struct DebuginfoTagDesc<'tcx> {
    pub ty: Ty<'tcx>,
    pub size: Size,
    pub align: Align,
    pub offset: Size,
}

#[derive(Debug)]
pub enum DebuginfoTypeDesc<'tcx> {
    /// Structs, tuples and closure environments.
    Struct {
        stub: DebuginfoStubDesc,
        members: Vec<DebuginfoMemberDesc<'tcx>>,
        generics: Vec<DebuginfoGenericParamDesc<'tcx>>,
    },
    ThinPointer {
        pointee: Ty<'tcx>,
        name: String,
        size: Size,
        align: Align,
    },
    /// `addr.ty` is unused: the address field always points to `pointee`.
    WidePointer {
        pointee: Ty<'tcx>,
        stub: DebuginfoStubDesc,
        addr: DebuginfoMemberDesc<'tcx>,
        extra: DebuginfoMemberDesc<'tcx>,
    },
    /// Enums with fields, in the native (non-MSVC) encoding.
    Enum {
        stub: DebuginfoStubDesc,
        variants: Vec<DebuginfoVariantDesc<'tcx>>,
        tag: Option<DebuginfoTagDesc<'tcx>>,
        variant_part_unique_id: String,
        generics: Vec<DebuginfoGenericParamDesc<'tcx>>,
    },
    /// The type is not handled by the shared path.
    Unsupported,
}

#[derive(Default)]
pub struct DebuginfoTypeDescCache<'tcx> {
    map: Lock<FxHashMap<Ty<'tcx>, Arc<DebuginfoTypeDesc<'tcx>>>>,
}

impl<'tcx> DebuginfoTypeDescCache<'tcx> {
    /// The lock is not held while `compute` runs, so `compute` may run more than once per type
    /// if codegen units race; the first result wins.
    pub fn get_or_compute(
        &self,
        ty: Ty<'tcx>,
        compute: impl FnOnce() -> DebuginfoTypeDesc<'tcx>,
    ) -> Arc<DebuginfoTypeDesc<'tcx>> {
        if let Some(desc) = self.map.lock().get(&ty) {
            return Arc::clone(desc);
        }
        let desc = Arc::new(compute());
        Arc::clone(self.map.lock().entry(ty).or_insert(desc))
    }
}
