//! Conversion of internal Rust compiler items to stable ones.

use rustc_abi::FieldIdx;
use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;

use super::Stable;
use crate::compiler_interface::BridgeTys;

mod abi;
mod mir;
mod ty;

impl<'tcx> Stable<'tcx> for rustc_hir::Safety {
    type T = crate::mir::Safety;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            rustc_hir::Safety::Unsafe => crate::mir::Safety::Unsafe,
            rustc_hir::Safety::Safe => crate::mir::Safety::Safe,
        }
    }
}

impl<'tcx> Stable<'tcx> for FieldIdx {
    type T = usize;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::CoroutineSource {
    type T = crate::mir::CoroutineSource;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_hir::CoroutineSource;
        match self {
            CoroutineSource::Block => crate::mir::CoroutineSource::Block,
            CoroutineSource::Closure => crate::mir::CoroutineSource::Closure,
            CoroutineSource::Fn => crate::mir::CoroutineSource::Fn,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::CoroutineKind {
    type T = crate::mir::CoroutineKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_hir::{CoroutineDesugaring, CoroutineKind};
        match *self {
            CoroutineKind::Desugared(CoroutineDesugaring::Async, source) => {
                crate::mir::CoroutineKind::Desugared(
                    crate::mir::CoroutineDesugaring::Async,
                    source.stable(tables, cx),
                )
            }
            CoroutineKind::Desugared(CoroutineDesugaring::Gen, source) => {
                crate::mir::CoroutineKind::Desugared(
                    crate::mir::CoroutineDesugaring::Gen,
                    source.stable(tables, cx),
                )
            }
            CoroutineKind::Coroutine(movability) => {
                crate::mir::CoroutineKind::Coroutine(movability.stable(tables, cx))
            }
            CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, source) => {
                crate::mir::CoroutineKind::Desugared(
                    crate::mir::CoroutineDesugaring::AsyncGen,
                    source.stable(tables, cx),
                )
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_span::Symbol {
    type T = crate::Symbol;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        self.to_string()
    }
}

impl<'tcx> Stable<'tcx> for rustc_span::Span {
    type T = crate::ty::Span;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        _: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        tables.create_span(*self)
    }
}
