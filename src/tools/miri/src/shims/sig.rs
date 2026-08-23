//! Everything related to checking the signature of shim invocations.

use rustc_abi::{CanonAbi, ExternAbi};
use rustc_middle::ty::{Binder, FnSig, FnSigKind, Ty};
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

/// Describes the expected signature of a shim.
pub struct ShimSig<'tcx, const ARGS: usize> {
    pub abi: ExternAbi,
    pub args: [Ty<'tcx>; ARGS],
    pub ret: Ty<'tcx>,
    pub c_variadic: bool,
}

/// Construct a `ShimSig` with convenient syntax:
/// ```rust,ignore
/// shim_sig!(extern "C" fn (*const T, i32) -> usize)
/// ```
///
/// The following types are supported:
/// - primitive integer types
/// - `()`
/// - (thin) raw pointers, written `*_` since the mutability and pointee type are irrelevant
/// - `$crate::$mod::...::$ty` for a type from the given crate (most commonly that is `libc`)
/// - `winapi::$ty` for a type from `std::sys::pal::windows::c`
#[macro_export]
macro_rules! shim_sig {
    (extern $abi:literal fn($($args:tt)*) -> $($ret:tt)*) => {
        |this| {
            let (args, c_variadic) = shim_sig_args_sep!(this, [$($args)*]);
            $crate::shims::sig::ShimSig {
                abi: std::str::FromStr::from_str($abi).expect("incorrect abi specified"),
                args,
                ret: shim_sig_arg!(this, $($ret)*),
                c_variadic,
            }
        }
    };
}

/// Computes a list of types for varargs, using the same syntax as `shim_sig!`.
#[macro_export]
macro_rules! shim_varargs {
    ($($args:tt)*) => {
        |this| {
            let (args, c_variadic) = shim_sig_args_sep!(this, [$($args)*]);
            assert!(!c_variadic); // don't accept `...` here
            args
        }
    };
}

/// Helper for `shim_sig!`.
///
/// Groups tokens into comma-separated chunks and calls the provided macro on them.
/// Returns a list of types and a boolean indicating whether there was a trailing `...`.
///
/// # Examples
///
/// ```ignore
/// shim_sig_args_sep!(this, [*_, i32, libc::off64_t]);
/// // expands to:
/// [shim_sig_arg!(*_), shim_sig_arg!(i32), shim_sig_arg!(libc::off64_t)];
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
    // No more tokens, trailing `...` - emit final output, indicate this is variadic.
    (@ $this:ident [$($final:tt)*] [...] ) => {
        ([$($final)*], true)
    };
    // No more tokens - emit final output, including final non-comma type.
    (@ $this:ident [$($final:tt)*] [$($collected:tt)+] ) => {
        ([$($final)* shim_sig_arg!($this, $($collected)*)], false)
    };
    // No more tokens, empty collector - emit final output.
    (@ $this:ident [$($final:tt)*] [] ) => {
        ([$($final)*], false)
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
    ($this:ident, !) => {
        $this.tcx.types.never
    };
    ($this:ident, bool) => {
        $this.tcx.types.bool
    };
    ($this:ident, *_) => {
        // Mutability does not matter for ABI.
        $this.machine.layouts.mut_raw_ptr.ty
    };
    ($this:ident, fn(..) -> _) => {
        // We currently treat fn ptrs as ABI-compatible with data ptrs so we can just use a raw ptr.
        $this.machine.layouts.const_raw_ptr.ty
    };
    ($this:ident, &[$($ty:tt)*]) => {
        rustc_middle::ty::Ty::new_ref(
            *$this.tcx,
            $this.tcx.lifetimes.re_erased,
            rustc_middle::ty::Ty::new_slice(*$this.tcx, shim_sig_arg!($this, $($ty)*)),
            rustc_middle::mir::Mutability::Not,
        )
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

impl<'tcx, const ARGS: usize> ShimSig<'tcx, ARGS> {
    fn as_abi(&self, ecx: &MiriInterpCx<'tcx>) -> &FnAbi<'tcx, Ty<'tcx>> {
        let mut inputs_and_output = Vec::with_capacity(ARGS.strict_add(1));
        inputs_and_output.extend(&self.args);
        inputs_and_output.push(self.ret);
        let fn_sig_binder = Binder::dummy(FnSig {
            inputs_and_output: ecx.machine.tcx.mk_type_list(&inputs_and_output),
            fn_sig_kind: FnSigKind::default().set_c_variadic(self.c_variadic).set_abi(self.abi),
        });
        ecx.fn_abi_of_fn_ptr(fn_sig_binder, Default::default()).unwrap()
    }
}

/// Helper function to compare two ABIs.
fn check_shim_abi<'tcx>(
    this: &MiriInterpCx<'tcx>,
    link_name: Symbol,
    callee_abi: &FnAbi<'tcx, Ty<'tcx>>,
    caller_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> InterpResult<'tcx> {
    if callee_abi.conv != caller_abi.conv {
        throw_ub_format!(
            r#"ABI mismatch: `{link_name}` has calling convention "{callee}", but the caller is using calling convention "{caller}""#,
            callee = callee_abi.conv,
            caller = caller_abi.conv,
        );
    }
    // No need to check unwinding: if the caller signature forbids unwinding, that's already
    // reflected in the unwind destination so if an unwind occurs it will be reported as UB.

    if caller_abi.c_variadic && !callee_abi.c_variadic {
        throw_ub_format!(
            "ABI mismatch: `{link_name}` is a non-variadic function, but the caller is using a variadic signature"
        );
    }
    if !caller_abi.c_variadic && callee_abi.c_variadic {
        throw_ub_format!(
            "ABI mismatch: `{link_name}` is a variadic function, but the caller is using a non-variadic signature"
        );
    }

    if callee_abi.fixed_count != caller_abi.fixed_count {
        throw_ub_format!(
            "ABI mismatch: calling `{link_name}` which takes {} {}argument{}, but {} argument{} given",
            callee_abi.fixed_count,
            if callee_abi.c_variadic { "fixed (non-variadic) " } else { "" },
            if callee_abi.fixed_count == 1 { "" } else { "s" },
            caller_abi.fixed_count,
            if caller_abi.fixed_count == 1 { " was" } else { "s were" },
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

/// Represents a tail of variadic arguments that have not yet been checked.
// Deliberately not `Copy` so that we don't consume the same vararg multiple times accidentally.
pub struct Varargs<'tcx, 'a> {
    args: &'a [OpTy<'tcx>],
    /// Number of variadic arguments that have already been taken, for error messages.
    already_gone: usize,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Ensure the given symbol is not exported by the program.
    fn check_shim_symbol_clash(&self, link_name: Symbol) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_ref();
        if let Some(instance) = this.lookup_exported_symbol(link_name)? {
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
                span: this.tcx.def_span(instance.def_id()).data(),
            })
        }
        interp_ok(())
    }

    /// 'Lenient' signature check. Deprecated; use `check_shim_sig` instead.
    fn check_shim_sig_deprecated<'a, const N: usize>(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
        link_name: Symbol,
        args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        self.check_shim_symbol_clash(link_name)?;

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
        &self,
        shim_sig: fn(&MiriInterpCx<'tcx>) -> ShimSig<'tcx, N>,
        // We take these as a tuple so that this takes less space on the caller side.
        (link_name, caller_fn_abi, caller_args): (Symbol, &FnAbi<'tcx, Ty<'tcx>>, &'a [OpTy<'tcx>]),
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        let this = self.eval_context_ref();

        // Compute callee ABI.
        let shim_sig = shim_sig(this);
        assert!(!shim_sig.c_variadic);
        let callee_fn_abi = shim_sig.as_abi(this);

        // Check everything.
        check_shim_abi(this, link_name, callee_fn_abi, caller_fn_abi)?;
        this.check_shim_symbol_clash(link_name)?;

        // Return arguments.
        if let Ok(ops) = caller_args.try_into() {
            return interp_ok(ops);
        }
        unreachable!()
    }

    /// Check that the given `caller_fn_abi` matches the expected ABI described by `shim_sig`, and
    /// then returns the list of fixed and variadic arguments in separate lists.
    fn check_shim_sig_variadic<'a, const N: usize>(
        &self,
        shim_sig: fn(&MiriInterpCx<'tcx>) -> ShimSig<'tcx, N>,
        // We take these as a tuple so that this takes less space on the caller side.
        (link_name, caller_fn_abi, caller_args): (Symbol, &FnAbi<'tcx, Ty<'tcx>>, &'a [OpTy<'tcx>]),
    ) -> InterpResult<'tcx, (&'a [OpTy<'tcx>; N], Varargs<'tcx, 'a>)> {
        let this = self.eval_context_ref();

        // Compute callee ABI.
        let shim_sig = shim_sig(this);
        assert!(shim_sig.c_variadic);
        let callee_fn_abi = shim_sig.as_abi(this);

        // Check everything.
        check_shim_abi(this, link_name, callee_fn_abi, caller_fn_abi)?;
        this.check_shim_symbol_clash(link_name)?;

        // Return arguments.
        if let Some((fixed, var)) = caller_args.split_first_chunk() {
            return interp_ok((fixed, Varargs { args: var, already_gone: 0 }));
        }
        unreachable!()
    }

    /// Fetches `N` arguments from `varargs`, checking their types.
    /// Also returns the remaining varargs.
    fn check_varargs<'a, const N: usize>(
        &self,
        tys: fn(&MiriInterpCx<'tcx>) -> [Ty<'tcx>; N],
        varargs: Varargs<'tcx, 'a>,
        fn_name: &str,
    ) -> InterpResult<'tcx, (&'a [OpTy<'tcx>; N], Varargs<'tcx, 'a>)> {
        let this = self.eval_context_ref();
        let tys = tys(this);

        let Some((now, tail)) = varargs.args.split_first_chunk::<N>() else {
            throw_ub_format!(
                "not enough variadic arguments for `{fn_name}`: got {}, expected at least {}",
                varargs.already_gone.strict_add(varargs.args.len()),
                varargs.already_gone.strict_add(N),
            )
        };

        for (n, (caller_gave, callee_expected)) in now.iter().zip(tys).enumerate() {
            // Check ABI compatibility. This is less strict than `next_arg` but we're also
            // not limited to just a few simple types.
            let callee_expected = this.layout_of(callee_expected)?;

            // FIXME: check compatibility once <https://github.com/rust-lang/rust/pull/161615>
            // landed.
            let _unused = (n, caller_gave, callee_expected);
        }

        interp_ok((now, Varargs { args: tail, already_gone: varargs.already_gone.strict_add(N) }))
    }

    /// Check that the given function has the expected amount of arguments, and then
    /// return the list of arguments.
    ///
    /// This may only be used for `extern "unadjusted"` LLVM intrinsics.
    fn check_shim_sig_unadjusted<'a, const N: usize>(
        &mut self,
        link_name: Symbol,
        args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        assert!(link_name.as_str().starts_with("llvm."));

        self.check_shim_symbol_clash(link_name)?;

        if let Ok(ops) = args.try_into() {
            return interp_ok(ops);
        }
        throw_ub_format!(
            "incorrect number of arguments for `{link_name}`: got {}, expected {}",
            args.len(),
            N
        )
    }
}
