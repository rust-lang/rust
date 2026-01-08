use std::iter;

use rustc_abi::ExternAbi;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::{self as hir, find_attr};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

use crate::lints::{ImproperGpuKernelArg, MissingGpuKernelExportName};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `improper_gpu_kernel_arg` lint detects incorrect use of types in `gpu-kernel`
    /// arguments.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-GPU targets)
    /// #[unsafe(no_mangle)]
    /// extern "gpu-kernel" fn kernel(_: [i32; 10]) {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: passing type `[i32; 10]` to a function with "gpu-kernel" ABI may have unexpected behavior
    ///  --> t.rs:2:34
    ///   |
    /// 2 | extern "gpu-kernel" fn kernel(_: [i32; 10]) {}
    ///   |                                  ^^^^^^^^^
    ///   |
    ///   = help: use primitive types and raw pointers to get reliable behavior
    ///   = note: `#[warn(improper_gpu_kernel_arg)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used as arguments in `gpu-kernel`
    /// functions follow certain rules to ensure proper compatibility with the foreign interfaces.
    /// This lint is issued when it detects a probable mistake in a signature.
    IMPROPER_GPU_KERNEL_ARG,
    Warn,
    "GPU kernel entry points have a limited ABI"
}

declare_lint! {
    /// The `missing_gpu_kernel_export_name` lint detects `gpu-kernel` functions that have a mangled name.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-GPU targets)
    /// extern "gpu-kernel" fn kernel() { }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: function with the "gpu-kernel" ABI has a mangled name
    ///  --> t.rs:1:1
    ///   |
    /// 1 | extern "gpu-kernel" fn kernel() {}
    ///   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ///   |
    ///   = help: use `unsafe(no_mangle)` or `unsafe(export_name = "<name>")`
    ///   = note: mangled names make it hard to find the kernel, this is usually not intended
    ///   = note: `#[warn(missing_gpu_kernel_export_name)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// `gpu-kernel` functions are usually searched by name in the compiled file.
    /// A mangled name is usually unintentional as it would need to be searched by the mangled name.
    ///
    /// To use an unmangled name for the kernel, either `no_mangle` or `export_name` can be used.
    /// ```rust,ignore (fails on non-GPU targets)
    /// // Can be found by the name "kernel"
    /// #[unsafe(no_mangle)]
    /// extern "gpu-kernel" fn kernel() { }
    ///
    /// // Can be found by the name "new_name"
    /// #[unsafe(export_name = "new_name")]
    /// extern "gpu-kernel" fn other_kernel() { }
    /// ```
    MISSING_GPU_KERNEL_EXPORT_NAME,
    Warn,
    "mangled gpu-kernel function"
}

declare_lint_pass!(ImproperGpuKernelLint => [
    IMPROPER_GPU_KERNEL_ARG,
    MISSING_GPU_KERNEL_EXPORT_NAME,
]);

/// Check for valid and invalid types.
struct CheckGpuKernelTypes<'tcx> {
    tcx: TyCtxt<'tcx>,
    // If one or more invalid types were encountered while folding.
    has_invalid: bool,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for CheckGpuKernelTypes<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match ty.kind() {
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {}
            // Thin pointers are allowed but fat pointers with metadata are not
            ty::RawPtr(_, _) => {
                if !ty.pointee_metadata_ty_or_projection(self.tcx).is_unit() {
                    self.has_invalid = true;
                }
            }

            ty::Adt(_, _)
            | ty::Alias(_, _)
            | ty::Array(_, _)
            | ty::Bound(_, _)
            | ty::Closure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::CoroutineWitness(..)
            | ty::Dynamic(_, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::Foreign(_)
            | ty::Never
            | ty::Pat(_, _)
            | ty::Placeholder(_)
            | ty::Ref(_, _, _)
            | ty::Slice(_)
            | ty::Str
            | ty::Tuple(_) => self.has_invalid = true,

            _ => return ty.super_fold_with(self),
        }
        ty
    }
}

/// `ImproperGpuKernelLint` checks `gpu-kernel` function definitions:
///
/// - `extern "gpu-kernel" fn` arguments should be primitive types.
/// - `extern "gpu-kernel" fn` should have an unmangled name.
impl<'tcx> LateLintPass<'tcx> for ImproperGpuKernelLint {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        _: &'tcx hir::Body<'_>,
        span: Span,
        id: LocalDefId,
    ) {
        use hir::intravisit::FnKind;

        let abi = match kind {
            FnKind::ItemFn(_, _, header, ..) => header.abi,
            FnKind::Method(_, sig, ..) => sig.header.abi,
            _ => return,
        };

        if abi != ExternAbi::GpuKernel {
            return;
        }

        let sig = cx.tcx.fn_sig(id).instantiate_identity();
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            let mut checker = CheckGpuKernelTypes { tcx: cx.tcx, has_invalid: false };
            input_ty.fold_with(&mut checker);
            if checker.has_invalid {
                cx.tcx.emit_node_span_lint(
                    IMPROPER_GPU_KERNEL_ARG,
                    input_hir.hir_id,
                    input_hir.span,
                    ImproperGpuKernelArg { ty: *input_ty },
                );
            }
        }

        // Check for no_mangle/export_name, so the kernel can be found when querying the compiled object for the kernel function by name
        if !find_attr!(
            cx.tcx.get_all_attrs(id),
            AttributeKind::NoMangle(..) | AttributeKind::ExportName { .. }
        ) {
            cx.emit_span_lint(MISSING_GPU_KERNEL_EXPORT_NAME, span, MissingGpuKernelExportName);
        }
    }
}
