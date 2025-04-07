//! Conversion of internal Rust compiler items to stable ones.

use rustc_abi::FieldIdx;

use crate::rustc_smir::{Stable, Tables};
use crate::stable_mir;

mod abi;
mod error;
mod mir;
mod ty;

impl<'tcx> Stable<'tcx> for rustc_hir::Safety {
    type T = stable_mir::mir::Safety;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_hir::Safety::Unsafe => stable_mir::mir::Safety::Unsafe,
            rustc_hir::Safety::Safe => stable_mir::mir::Safety::Safe,
        }
    }
}

impl<'tcx> Stable<'tcx> for FieldIdx {
    type T = usize;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::CoroutineSource {
    type T = stable_mir::mir::CoroutineSource;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use rustc_hir::CoroutineSource;
        match self {
            CoroutineSource::Block => stable_mir::mir::CoroutineSource::Block,
            CoroutineSource::Closure => stable_mir::mir::CoroutineSource::Closure,
            CoroutineSource::Fn => stable_mir::mir::CoroutineSource::Fn,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::CoroutineKind {
    type T = stable_mir::mir::CoroutineKind;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_hir::{CoroutineDesugaring, CoroutineKind};
        match *self {
            CoroutineKind::Desugared(CoroutineDesugaring::Async, source) => {
                stable_mir::mir::CoroutineKind::Desugared(
                    stable_mir::mir::CoroutineDesugaring::Async,
                    source.stable(tables),
                )
            }
            CoroutineKind::Desugared(CoroutineDesugaring::Gen, source) => {
                stable_mir::mir::CoroutineKind::Desugared(
                    stable_mir::mir::CoroutineDesugaring::Gen,
                    source.stable(tables),
                )
            }
            CoroutineKind::Coroutine(movability) => {
                stable_mir::mir::CoroutineKind::Coroutine(movability.stable(tables))
            }
            CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, source) => {
                stable_mir::mir::CoroutineKind::Desugared(
                    stable_mir::mir::CoroutineDesugaring::AsyncGen,
                    source.stable(tables),
                )
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_span::Symbol {
    type T = stable_mir::Symbol;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        self.to_string()
    }
}

impl<'tcx> Stable<'tcx> for rustc_span::Span {
    type T = stable_mir::ty::Span;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        tables.create_span(*self)
    }
}
