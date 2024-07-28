use std::cell::RefCell;

use rustc_data_structures::{
    fingerprint::Fingerprint,
    fx::FxHashMap,
    stable_hasher::{HashStable, StableHasher},
};
use rustc_macros::HashStable;
use rustc_middle::{
    bug,
    ty::{ParamEnv, PolyExistentialTraitRef, Ty, TyCtxt},
};
use rustc_target::abi::{Align, Size, VariantIdx};

use crate::{
    common::CodegenCx,
    debuginfo::utils::{create_DIArray, debug_context, DIB},
    llvm::{
        self,
        debuginfo::{DIFlags, DIScope, DIType},
    },
};

use super::{unknown_file_metadata, SmallVec, UNKNOWN_LINE_NUMBER};

mod private {
    use rustc_macros::HashStable;

    // This type cannot be constructed outside of this module because
    // it has a private field. We make use of this in order to prevent
    // `UniqueTypeId` from being constructed directly, without asserting
    // the preconditions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, HashStable)]
    pub struct HiddenZst;
}

/// A unique identifier for anything that we create a debuginfo node for.
/// The types it contains are expected to already be normalized (which
/// is asserted in the constructors).
///
/// Note that there are some things that only show up in debuginfo, like
/// the separate type descriptions for each enum variant. These get an ID
/// too because they have their own debuginfo node in LLVM IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, HashStable)]
pub(super) enum UniqueTypeId<'tcx> {
    /// The ID of a regular type as it shows up at the language level.
    Ty(Ty<'tcx>, private::HiddenZst),
    /// The ID for the single DW_TAG_variant_part nested inside the top-level
    /// DW_TAG_structure_type that describes enums and coroutines.
    VariantPart(Ty<'tcx>, private::HiddenZst),
    /// The ID for the artificial struct type describing a single enum variant.
    VariantStructType(Ty<'tcx>, VariantIdx, private::HiddenZst),
    /// The ID for the additional wrapper struct type describing an enum variant in CPP-like mode.
    VariantStructTypeCppLikeWrapper(Ty<'tcx>, VariantIdx, private::HiddenZst),
    /// The ID of the artificial type we create for VTables.
    VTableTy(Ty<'tcx>, Option<PolyExistentialTraitRef<'tcx>>, private::HiddenZst),
}

impl<'tcx> UniqueTypeId<'tcx> {
    pub fn for_ty(tcx: TyCtxt<'tcx>, t: Ty<'tcx>) -> Self {
        assert_eq!(t, tcx.normalize_erasing_regions(ParamEnv::reveal_all(), t));
        UniqueTypeId::Ty(t, private::HiddenZst)
    }

    pub fn for_enum_variant_part(tcx: TyCtxt<'tcx>, enum_ty: Ty<'tcx>) -> Self {
        assert_eq!(enum_ty, tcx.normalize_erasing_regions(ParamEnv::reveal_all(), enum_ty));
        UniqueTypeId::VariantPart(enum_ty, private::HiddenZst)
    }

    pub fn for_enum_variant_struct_type(
        tcx: TyCtxt<'tcx>,
        enum_ty: Ty<'tcx>,
        variant_idx: VariantIdx,
    ) -> Self {
        assert_eq!(enum_ty, tcx.normalize_erasing_regions(ParamEnv::reveal_all(), enum_ty));
        UniqueTypeId::VariantStructType(enum_ty, variant_idx, private::HiddenZst)
    }

    pub fn for_enum_variant_struct_type_wrapper(
        tcx: TyCtxt<'tcx>,
        enum_ty: Ty<'tcx>,
        variant_idx: VariantIdx,
    ) -> Self {
        assert_eq!(enum_ty, tcx.normalize_erasing_regions(ParamEnv::reveal_all(), enum_ty));
        UniqueTypeId::VariantStructTypeCppLikeWrapper(enum_ty, variant_idx, private::HiddenZst)
    }

    pub fn for_vtable_ty(
        tcx: TyCtxt<'tcx>,
        self_type: Ty<'tcx>,
        implemented_trait: Option<PolyExistentialTraitRef<'tcx>>,
    ) -> Self {
        assert_eq!(self_type, tcx.normalize_erasing_regions(ParamEnv::reveal_all(), self_type));
        assert_eq!(
            implemented_trait,
            tcx.normalize_erasing_regions(ParamEnv::reveal_all(), implemented_trait)
        );
        UniqueTypeId::VTableTy(self_type, implemented_trait, private::HiddenZst)
    }

    /// Generates a string version of this [UniqueTypeId], which can be used as the `UniqueId`
    /// argument of the various `LLVMRustDIBuilderCreate*Type()` methods.
    ///
    /// Right now this takes the form of a hex-encoded opaque hash value.
    pub fn generate_unique_id_string(self, tcx: TyCtxt<'tcx>) -> String {
        let mut hasher = StableHasher::new();
        tcx.with_stable_hashing_context(|mut hcx| {
            hcx.while_hashing_spans(false, |hcx| self.hash_stable(hcx, &mut hasher))
        });
        hasher.finish::<Fingerprint>().to_hex()
    }

    pub fn expect_ty(self) -> Ty<'tcx> {
        match self {
            UniqueTypeId::Ty(ty, _) => ty,
            _ => bug!("Expected `UniqueTypeId::Ty` but found `{:?}`", self),
        }
    }
}

/// The `TypeMap` is where the debug context holds the type metadata nodes
/// created so far. The debuginfo nodes are identified by `UniqueTypeId`.
#[derive(Default)]
pub(crate) struct TypeMap<'ll, 'tcx> {
    pub(super) unique_id_to_di_node: RefCell<FxHashMap<UniqueTypeId<'tcx>, &'ll DIType>>,
}

impl<'ll, 'tcx> TypeMap<'ll, 'tcx> {
    /// Adds a `UniqueTypeId` to metadata mapping to the `TypeMap`. The method will
    /// fail if the mapping already exists.
    pub(super) fn insert(&self, unique_type_id: UniqueTypeId<'tcx>, metadata: &'ll DIType) {
        if self.unique_id_to_di_node.borrow_mut().insert(unique_type_id, metadata).is_some() {
            bug!("type metadata for unique ID '{:?}' is already in the `TypeMap`!", unique_type_id);
        }
    }

    pub(super) fn di_node_for_unique_id(
        &self,
        unique_type_id: UniqueTypeId<'tcx>,
    ) -> Option<&'ll DIType> {
        self.unique_id_to_di_node.borrow().get(&unique_type_id).cloned()
    }
}

pub struct DINodeCreationResult<'ll> {
    pub di_node: &'ll DIType,
    pub already_stored_in_typemap: bool,
}

impl<'ll> DINodeCreationResult<'ll> {
    pub fn new(di_node: &'ll DIType, already_stored_in_typemap: bool) -> Self {
        DINodeCreationResult { di_node, already_stored_in_typemap }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Stub<'ll> {
    Struct,
    Union,
    VTableTy { vtable_holder: &'ll DIType },
}

pub struct StubInfo<'ll, 'tcx> {
    metadata: &'ll DIType,
    unique_type_id: UniqueTypeId<'tcx>,
}

impl<'ll, 'tcx> StubInfo<'ll, 'tcx> {
    pub(super) fn new(
        cx: &CodegenCx<'ll, 'tcx>,
        unique_type_id: UniqueTypeId<'tcx>,
        build: impl FnOnce(&CodegenCx<'ll, 'tcx>, /* unique_type_id_str: */ &str) -> &'ll DIType,
    ) -> StubInfo<'ll, 'tcx> {
        let unique_type_id_str = unique_type_id.generate_unique_id_string(cx.tcx);
        let di_node = build(cx, &unique_type_id_str);
        StubInfo { metadata: di_node, unique_type_id }
    }
}

/// Create a stub debuginfo node onto which fields and nested types can be attached.
pub(super) fn stub<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    kind: Stub<'ll>,
    unique_type_id: UniqueTypeId<'tcx>,
    name: &str,
    (size, align): (Size, Align),
    containing_scope: Option<&'ll DIScope>,
    flags: DIFlags,
) -> StubInfo<'ll, 'tcx> {
    let empty_array = create_DIArray(DIB(cx), &[]);
    let unique_type_id_str = unique_type_id.generate_unique_id_string(cx.tcx);

    let metadata = match kind {
        Stub::Struct | Stub::VTableTy { .. } => {
            let vtable_holder = match kind {
                Stub::VTableTy { vtable_holder } => Some(vtable_holder),
                _ => None,
            };
            unsafe {
                llvm::LLVMRustDIBuilderCreateStructType(
                    DIB(cx),
                    containing_scope,
                    name.as_ptr().cast(),
                    name.len(),
                    unknown_file_metadata(cx),
                    UNKNOWN_LINE_NUMBER,
                    size.bits(),
                    align.bits() as u32,
                    flags,
                    None,
                    empty_array,
                    0,
                    vtable_holder,
                    unique_type_id_str.as_ptr().cast(),
                    unique_type_id_str.len(),
                )
            }
        }
        Stub::Union => unsafe {
            llvm::LLVMRustDIBuilderCreateUnionType(
                DIB(cx),
                containing_scope,
                name.as_ptr().cast(),
                name.len(),
                unknown_file_metadata(cx),
                UNKNOWN_LINE_NUMBER,
                size.bits(),
                align.bits() as u32,
                flags,
                Some(empty_array),
                0,
                unique_type_id_str.as_ptr().cast(),
                unique_type_id_str.len(),
            )
        },
    };
    StubInfo { metadata, unique_type_id }
}

/// This function enables creating debuginfo nodes that can recursively refer to themselves.
/// It will first insert the given stub into the type map and only then execute the `members`
/// and `generics` closures passed in. These closures have access to the stub so they can
/// directly attach fields to them. If the type of a field transitively refers back
/// to the type currently being built, the stub will already be found in the type map,
/// which effectively breaks the recursion cycle.
pub(super) fn build_type_with_children<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    stub_info: StubInfo<'ll, 'tcx>,
    members: impl FnOnce(&CodegenCx<'ll, 'tcx>, &'ll DIType) -> SmallVec<&'ll DIType>,
    generics: impl FnOnce(&CodegenCx<'ll, 'tcx>) -> SmallVec<&'ll DIType>,
) -> DINodeCreationResult<'ll> {
    assert_eq!(debug_context(cx).type_map.di_node_for_unique_id(stub_info.unique_type_id), None);

    debug_context(cx).type_map.insert(stub_info.unique_type_id, stub_info.metadata);

    let members: SmallVec<_> =
        members(cx, stub_info.metadata).into_iter().map(|node| Some(node)).collect();
    let generics: SmallVec<Option<&'ll DIType>> =
        generics(cx).into_iter().map(|node| Some(node)).collect();

    if !(members.is_empty() && generics.is_empty()) {
        unsafe {
            let members_array = create_DIArray(DIB(cx), &members[..]);
            let generics_array = create_DIArray(DIB(cx), &generics[..]);
            llvm::LLVMRustDICompositeTypeReplaceArrays(
                DIB(cx),
                stub_info.metadata,
                Some(members_array),
                Some(generics_array),
            );
        }
    }

    DINodeCreationResult { di_node: stub_info.metadata, already_stored_in_typemap: true }
}
