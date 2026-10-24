//! Implementation of CompilerCtxt.

#![allow(rustc::usage_of_qualified_ty)]

use std::marker::PhantomData;

use rustc_abi::HasDataLayout;
use rustc_middle::ty;
use rustc_middle::ty::layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers};
use rustc_middle::ty::{Ty, TyCtxt};

use crate::{Bridge, Error};

mod helpers;
mod impls;

pub use helpers::*;

/// Provides direct access to rustc's internal queries.
///
/// `CompilerInterface` must go through
/// this context to obtain internal information.
pub struct CompilerCtxt<'tcx, B: Bridge> {
    pub tcx: TyCtxt<'tcx>,
    _marker: PhantomData<B>,
}

impl<'tcx, B: Bridge> CompilerCtxt<'tcx, B> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, _marker: Default::default() }
    }
}

/// Implement error handling for extracting function ABI information.
impl<'tcx, B: Bridge> FnAbiOfHelpers<'tcx> for CompilerCtxt<'tcx, B> {
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

impl<'tcx, B: Bridge> LayoutOfHelpers<'tcx> for CompilerCtxt<'tcx, B> {
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

impl<'tcx, B: Bridge> HasTypingEnv<'tcx> for CompilerCtxt<'tcx, B> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx, B: Bridge> HasTyCtxt<'tcx> for CompilerCtxt<'tcx, B> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx, B: Bridge> HasDataLayout for CompilerCtxt<'tcx, B> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.tcx.data_layout()
    }
}
