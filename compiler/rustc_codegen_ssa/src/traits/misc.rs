use std::cell::RefCell;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::Session;

use super::BackendTypes;

/// Strategy for incorporating address-based diversity into PAC computation.
pub enum AddressDiversity {
    /// No address diversity is applied.
    None,
    /// Use the actual memory address for diversification.
    Real,
    /// Use a fixed synthetic value instead of the real address,
    /// i.e. `1` is used for `.init_array` / `.fini_array`.
    Synthetic(u64),
}

impl Default for AddressDiversity {
    fn default() -> Self {
        AddressDiversity::None
    }
}

/// Metadata used for pointer authentication.
pub struct PacMetadata {
    /// The PAC key to use.
    pub key: u32,
    /// Discriminator value used to diversify the PAC.
    pub disc: u64,
    /// Controls how address diversity is applied when computing the PAC.
    pub addr_diversity: AddressDiversity,
}

impl Default for PacMetadata {
    fn default() -> Self {
        PacMetadata { key: 0, disc: 0, addr_diversity: AddressDiversity::default() }
    }
}

pub trait MiscCodegenMethods<'tcx>: BackendTypes {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), Self::Value>>;
    fn apply_vcall_visibility_metadata(
        &self,
        _ty: Ty<'tcx>,
        _poly_trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
        _vtable: Self::Value,
    ) {
    }
    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Function;
    fn get_fn_addr(&self, instance: Instance<'tcx>, pac: Option<PacMetadata>) -> Self::Value;
    fn eh_personality(&self) -> Self::Function;
    fn sess(&self) -> &Session;
    fn set_frame_pointer_type(&self, llfn: Self::Function);
    fn apply_target_cpu_attr(&self, llfn: Self::Function);
    /// Declares the extern "C" main function for the entry point. Returns None if the symbol
    /// already exists.
    fn declare_c_main(&self, fn_type: Self::FunctionSignature) -> Option<Self::Function>;
}
