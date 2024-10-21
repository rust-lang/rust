//! Provides the `extern static` that this platform expects.

use crate::*;

impl<'tcx> MiriMachine<'tcx> {
    fn alloc_extern_static(
        this: &mut MiriInterpCx<'tcx>,
        name: &str,
        val: ImmTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let place = this.allocate(val.layout, MiriMemoryKind::ExternStatic.into())?;
        this.write_immediate(*val, &place)?;
        Self::add_extern_static(this, name, place.ptr());
        interp_ok(())
    }

    /// Zero-initialized pointer-sized extern statics are pretty common.
    /// Most of them are for weak symbols, which we all set to null (indicating that the
    /// symbol is not supported, and triggering fallback code which ends up calling
    /// some other shim that we do support).
    fn null_ptr_extern_statics(
        this: &mut MiriInterpCx<'tcx>,
        names: &[&str],
    ) -> InterpResult<'tcx> {
        for name in names {
            let val = ImmTy::from_int(0, this.machine.layouts.usize);
            Self::alloc_extern_static(this, name, val)?;
        }
        interp_ok(())
    }

    /// Extern statics that are initialized with function pointers to the symbols of the same name.
    fn weak_symbol_extern_statics(
        this: &mut MiriInterpCx<'tcx>,
        names: &[&str],
    ) -> InterpResult<'tcx> {
        for name in names {
            assert!(this.is_dyn_sym(name), "{name} is not a dynamic symbol");
            let layout = this.machine.layouts.const_raw_ptr;
            let ptr = this.fn_ptr(FnVal::Other(DynSym::from_str(name)));
            let val = ImmTy::from_scalar(Scalar::from_pointer(ptr, this), layout);
            Self::alloc_extern_static(this, name, val)?;
        }
        interp_ok(())
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics(this: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        // "__rust_no_alloc_shim_is_unstable"
        let val = ImmTy::from_int(0, this.machine.layouts.u8); // always 0, value does not matter
        Self::alloc_extern_static(this, "__rust_no_alloc_shim_is_unstable", val)?;

        // "__rust_alloc_error_handler_should_panic"
        let val = this.tcx.sess.opts.unstable_opts.oom.should_panic();
        let val = ImmTy::from_int(val, this.machine.layouts.u8);
        Self::alloc_extern_static(this, "__rust_alloc_error_handler_should_panic", val)?;

        if this.target_os_is_unix() {
            // "environ" is mandated by POSIX.
            let environ = this.machine.env_vars.unix().environ();
            Self::add_extern_static(this, "environ", environ);
        }

        match this.tcx.sess.target.os.as_ref() {
            "linux" => {
                Self::null_ptr_extern_statics(this, &[
                    "__cxa_thread_atexit_impl",
                    "__clock_gettime64",
                ])?;
                Self::weak_symbol_extern_statics(this, &["getrandom", "statx"])?;
            }
            "freebsd" => {
                Self::null_ptr_extern_statics(this, &["__cxa_thread_atexit_impl"])?;
            }
            "android" => {
                Self::null_ptr_extern_statics(this, &["bsd_signal"])?;
                Self::weak_symbol_extern_statics(this, &["signal", "getrandom"])?;
            }
            "windows" => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let val = ImmTy::from_int(0, this.machine.layouts.u8);
                Self::alloc_extern_static(this, "_tls_used", val)?;
            }
            "illumos" | "solaris" => {
                Self::weak_symbol_extern_statics(this, &["pthread_setname_np"])?;
            }
            _ => {} // No "extern statics" supported on this target
        }
        interp_ok(())
    }
}
