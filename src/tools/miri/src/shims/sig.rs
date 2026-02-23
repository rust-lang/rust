//! Everything related to checking the signature of shim invocations.

use rustc_abi::{CanonAbi, ExternAbi};
use rustc_hir::Safety;
use rustc_middle::ty::{Binder, FnSig, Ty};
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

/// Describes the expected signature of a shim.
pub struct ShimSig<'tcx, const ARGS: usize> {
    pub abi: ExternAbi,
    pub args: [Ty<'tcx>; ARGS],
    pub ret: Ty<'tcx>,
}

/// Construct a `ShimSig` with convenient syntax:
/// ```rust,ignore
/// shim_sig!(extern "C" fn (*const T, i32) -> usize)
/// ```
///
/// The following types are supported:
/// - primitive integer types
/// - `()`
/// - (thin) raw pointers, written `*const _` and `*mut _` since the pointee type is irrelevant
/// - `$crate::$mod::...::$ty` for a type from the given crate (most commonly that is `libc`)
/// - `winapi::$ty` for a type from `std::sys::pal::windows::c`
#[macro_export]
macro_rules! shim_sig {
    (extern $abi:literal fn($($args:tt)*) -> $($ret:tt)*) => {
        |this| $crate::shims::sig::ShimSig {
            abi: std::str::FromStr::from_str($abi).expect("incorrect abi specified"),
            args: shim_sig_args_sep!(this, [$($args)*]),
            ret: shim_sig_arg!(this, $($ret)*),
        }
    };
}

/// Helper for `shim_sig!`.
///
/// Groups tokens into comma-separated chunks and calls the provided macro on them.
///
/// # Examples
///
/// ```ignore
/// shim_sig_args_sep!(this, [*const _, i32, libc::off64_t]);
/// // expands to:
/// [shim_sig_arg!(*const _), shim_sig_arg!(i32), shim_sig_arg!(libc::off64_t)];
/// ```
#[macro_export]
macro_rules! shim_sig_args_sep {
    ($this:ident, [$($tt:tt)*]) => {
        shim_sig_args_sep!(@ $this [] [] $($tt)*)
    };

    // All below matchers form a fairly simple iterator over the input.
    // - Non-comma token - append to collector
    // - Comma token - call the provided macro on the collector and reset the collector
    // - End of input - empty collector one last time. emit output as an array

    // Handles `,` token - take collected type and call shim_sig_arg on it.
    // Append the result to the final output.
    (@ $this:ident [$($final:tt)*] [$($collected:tt)*] , $($tt:tt)*) => {
        shim_sig_args_sep!(@ $this [$($final)* shim_sig_arg!($this, $($collected)*), ] [] $($tt)*)
    };
    // Handle non-comma token - append to collected type.
    (@ $this:ident [$($final:tt)*] [$($collected:tt)*] $first:tt $($tt:tt)*) => {
        shim_sig_args_sep!(@ $this [$($final)*] [$($collected)* $first] $($tt)*)
    };
    // No more tokens - emit final output, including final non-comma type.
    (@ $this:ident [$($final:tt)*] [$($collected:tt)+] ) => {
        [$($final)* shim_sig_arg!($this, $($collected)*)]
    };
    // No more tokens - emit final output.
    (@ $this:ident [$($final:tt)*] [] ) => {
        [$($final)*]
    };
}

/// Helper for `shim_sig!`.
///
/// Converts a type
#[macro_export]
macro_rules! shim_sig_arg {
    ($this:ident, i8) => {
        $this.tcx.types.i8
    };
    ($this:ident, i16) => {
        $this.tcx.types.i16
    };
    ($this:ident, i32) => {
        $this.tcx.types.i32
    };
    ($this:ident, i64) => {
        $this.tcx.types.i64
    };
    ($this:ident, i128) => {
        $this.tcx.types.i128
    };
    ($this:ident, isize) => {
        $this.tcx.types.isize
    };
    ($this:ident, u8) => {
        $this.tcx.types.u8
    };
    ($this:ident, u16) => {
        $this.tcx.types.u16
    };
    ($this:ident, u32) => {
        $this.tcx.types.u32
    };
    ($this:ident, u64) => {
        $this.tcx.types.u64
    };
    ($this:ident, u128) => {
        $this.tcx.types.u128
    };
    ($this:ident, usize) => {
        $this.tcx.types.usize
    };
    ($this:ident, ()) => {
        $this.tcx.types.unit
    };
    ($this:ident, bool) => {
        $this.tcx.types.bool
    };
    ($this:ident, *const _) => {
        $this.machine.layouts.const_raw_ptr.ty
    };
    ($this:ident, *mut _) => {
        $this.machine.layouts.mut_raw_ptr.ty
    };
    ($this:ident, winapi::$ty:ident) => {
        $this.windows_ty_layout(stringify!($ty)).ty
    };
    ($this:ident, $krate:ident :: $($path:ident)::+) => {
        helpers::path_ty_layout($this, &[stringify!($krate), $(stringify!($path)),*]).ty
    };
    ($this:ident, $($other:tt)*) => {
        compile_error!(concat!("unsupported signature type: ", stringify!($($other)*)))
    }
}

/// Helper function to compare two ABIs.
fn check_shim_abi<'tcx>(
    this: &MiriInterpCx<'tcx>,
    callee_abi: &FnAbi<'tcx, Ty<'tcx>>,
    caller_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> InterpResult<'tcx> {
    if callee_abi.conv != caller_abi.conv {
        throw_ub_format!(
            r#"calling a function with calling convention "{callee}" using caller calling convention "{caller}""#,
            callee = callee_abi.conv,
            caller = caller_abi.conv,
        );
    }
    if callee_abi.can_unwind && !caller_abi.can_unwind {
        throw_ub_format!(
            "ABI mismatch: callee may unwind, but caller-side signature prohibits unwinding",
        );
    }
    if caller_abi.c_variadic && !callee_abi.c_variadic {
        throw_ub_format!(
            "ABI mismatch: calling a non-variadic function with a variadic caller-side signature"
        );
    }
    if !caller_abi.c_variadic && callee_abi.c_variadic {
        throw_ub_format!(
            "ABI mismatch: calling a variadic function with a non-variadic caller-side signature"
        );
    }

    if callee_abi.fixed_count != caller_abi.fixed_count {
        throw_ub_format!(
            "ABI mismatch: expected {} arguments, found {} arguments ",
            callee_abi.fixed_count,
            caller_abi.fixed_count
        );
    }

    if !this.check_argument_compat(&caller_abi.ret, &callee_abi.ret)? {
        throw_ub!(AbiMismatchReturn {
            caller_ty: caller_abi.ret.layout.ty,
            callee_ty: callee_abi.ret.layout.ty
        });
    }

    for (idx, (caller_arg, callee_arg)) in
        caller_abi.args.iter().zip(callee_abi.args.iter()).enumerate()
    {
        if !this.check_argument_compat(caller_arg, callee_arg)? {
            throw_ub!(AbiMismatchArgument {
                arg_idx: idx,
                caller_ty: caller_abi.args[idx].layout.ty,
                callee_ty: callee_abi.args[idx].layout.ty
            });
        }
    }

    interp_ok(())
}

fn check_shim_symbol_clash<'tcx>(
    this: &mut MiriInterpCx<'tcx>,
    link_name: Symbol,
) -> InterpResult<'tcx, ()> {
    if let Some((body, instance)) = this.lookup_exported_symbol(link_name)? {
        // If compiler-builtins is providing the symbol, then don't treat it as a clash.
        // We'll use our built-in implementation in `emulate_foreign_item_inner` for increased
        // performance. Note that this means we won't catch any undefined behavior in
        // compiler-builtins when running other crates, but Miri can still be run on
        // compiler-builtins itself (or any crate that uses it as a normal dependency)
        if this.tcx.is_compiler_builtins(instance.def_id().krate) {
            return interp_ok(());
        }

        throw_machine_stop!(TerminationInfo::SymbolShimClashing {
            link_name,
            span: body.span.data(),
        })
    }
    interp_ok(())
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn check_shim_sig_lenient<'a, const N: usize>(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
        link_name: Symbol,
        args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        let this = self.eval_context_mut();
        check_shim_symbol_clash(this, link_name)?;

        if abi.conv != exp_abi {
            throw_ub_format!(
                r#"calling a function with calling convention "{exp_abi}" using caller calling convention "{}""#,
                abi.conv
            );
        }
        if abi.c_variadic {
            throw_ub_format!(
                "calling a non-variadic function with a variadic caller-side signature"
            );
        }

        if let Ok(ops) = args.try_into() {
            return interp_ok(ops);
        }
        throw_ub_format!(
            "incorrect number of arguments for `{link_name}`: got {}, expected {}",
            args.len(),
            N
        )
    }

    /// Check that the given `caller_fn_abi` matches the expected ABI described by `shim_sig`, and
    /// then returns the list of arguments.
    fn check_shim_sig<'a, const N: usize>(
        &mut self,
        shim_sig: fn(&MiriInterpCx<'tcx>) -> ShimSig<'tcx, N>,
        link_name: Symbol,
        caller_fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        caller_args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        let this = self.eval_context_mut();
        let shim_sig = shim_sig(this);

        // Compute full callee ABI.
        let mut inputs_and_output = Vec::with_capacity(N.strict_add(1));
        inputs_and_output.extend(&shim_sig.args);
        inputs_and_output.push(shim_sig.ret);
        let fn_sig_binder = Binder::dummy(FnSig {
            inputs_and_output: this.machine.tcx.mk_type_list(&inputs_and_output),
            c_variadic: false,
            // This does not matter for the ABI.
            safety: Safety::Safe,
            abi: shim_sig.abi,
        });
        let callee_fn_abi = this.fn_abi_of_fn_ptr(fn_sig_binder, Default::default())?;

        // Check everything.
        check_shim_abi(this, callee_fn_abi, caller_fn_abi)?;
        check_shim_symbol_clash(this, link_name)?;

        // Return arguments.
        if let Ok(ops) = caller_args.try_into() {
            return interp_ok(ops);
        }
        unreachable!()
    }

    /// Check shim for variadic function.
    /// Returns a tuple that consisting of an array of fixed args, and a slice of varargs.
    fn check_shim_sig_variadic_lenient<'a, const N: usize>(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
        link_name: Symbol,
        args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, (&'a [OpTy<'tcx>; N], &'a [OpTy<'tcx>])>
    where
        &'a [OpTy<'tcx>; N]: TryFrom<&'a [OpTy<'tcx>]>,
    {
        let this = self.eval_context_mut();
        check_shim_symbol_clash(this, link_name)?;

        if abi.conv != exp_abi {
            throw_ub_format!(
                r#"calling a function with calling convention "{exp_abi}" using caller calling convention "{}""#,
                abi.conv
            );
        }
        if !abi.c_variadic {
            throw_ub_format!(
                "calling a variadic function with a non-variadic caller-side signature"
            );
        }
        if abi.fixed_count != u32::try_from(N).unwrap() {
            throw_ub_format!(
                "incorrect number of fixed arguments for variadic function `{}`: got {}, expected {N}",
                link_name.as_str(),
                abi.fixed_count
            )
        }
        if let Some(args) = args.split_first_chunk() {
            return interp_ok(args);
        }
        panic!("mismatch between signature and `args` slice");
    }
}

/// Check that the number of varargs is at least the minimum what we expect.
/// Fixed args should not be included.
pub fn check_min_vararg_count<'a, 'tcx, const N: usize>(
    name: &'a str,
    args: &'a [OpTy<'tcx>],
) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
    if let Some((ops, _)) = args.split_first_chunk() {
        return interp_ok(ops);
    }
    throw_ub_format!(
        "not enough variadic arguments for `{name}`: got {}, expected at least {}",
        args.len(),
        N
    )
}
