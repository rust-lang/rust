//! Implementation of StableMIR Context.

#![allow(rustc::usage_of_qualified_ty)]

use std::marker::PhantomData;

use rustc_abi::HasDataLayout;
use rustc_middle::ty;
use rustc_middle::ty::layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers};
use rustc_middle::ty::{Ty, TyCtxt};

use crate::rustc_smir::{Bridge, SmirError, Tables};

mod impls;
mod traits;

pub use traits::*;

/// Provides direct access to rustc's internal queries.
///
/// The [`crate::stable_mir::compiler_interface::SmirInterface`] must go through
/// this context to obtain rustc-level information.
pub struct SmirCtxt<'tcx, B: Bridge> {
    tcx: TyCtxt<'tcx>,
    _marker: PhantomData<B>,
}

impl<'tcx, B: Bridge> SmirCtxt<'tcx, B> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, _marker: Default::default() }
    }
}

/// Implement error handling for extracting function ABI information.
impl<'tcx, B: Bridge> FnAbiOfHelpers<'tcx> for Tables<'tcx, B> {
    type FnAbiOfResult = Result<&'tcx rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>, B::Error>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: ty::layout::FnAbiError<'tcx>,
        _span: rustc_span::Span,
        fn_abi_request: ty::layout::FnAbiRequest<'tcx>,
    ) -> B::Error {
        B::Error::new(format!("Failed to get ABI for `{fn_abi_request:?}`: {err:?}"))
    }
}

impl<'tcx, B: Bridge> LayoutOfHelpers<'tcx> for Tables<'tcx, B> {
    type LayoutOfResult = Result<ty::layout::TyAndLayout<'tcx>, B::Error>;

    #[inline]
    fn handle_layout_err(
        &self,
        err: ty::layout::LayoutError<'tcx>,
        _span: rustc_span::Span,
        ty: Ty<'tcx>,
    ) -> B::Error {
        B::Error::new(format!("Failed to get layout for `{ty}`: {err}"))
    }
}

impl<'tcx, B: Bridge> HasTypingEnv<'tcx> for Tables<'tcx, B> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx, B: Bridge> HasTyCtxt<'tcx> for Tables<'tcx, B> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx, B: Bridge> HasDataLayout for Tables<'tcx, B> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.tcx.data_layout()
    }
}
