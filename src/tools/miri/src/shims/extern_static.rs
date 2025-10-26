//! Provides the `extern static` that this platform expects.

use crate::*;

impl<'tcx> MiriMachine<'tcx> {
    fn alloc_extern_static(
        ecx: &mut MiriInterpCx<'tcx>,
        name: &str,
        val: ImmTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let place = ecx.allocate(val.layout, MiriMemoryKind::ExternStatic.into())?;
        ecx.write_immediate(*val, &place)?;
        Self::add_extern_static(ecx, name, place.ptr());
        interp_ok(())
    }

    /// Zero-initialized pointer-sized extern statics are pretty common.
    /// Most of them are for weak symbols, which we all set to null (indicating that the
    /// symbol is not supported, and triggering fallback code which ends up calling
    /// some other shim that we do support).
    fn null_ptr_extern_statics(ecx: &mut MiriInterpCx<'tcx>, names: &[&str]) -> InterpResult<'tcx> {
        for name in names {
            let val = ImmTy::from_int(0, ecx.machine.layouts.usize);
            Self::alloc_extern_static(ecx, name, val)?;
        }
        interp_ok(())
    }

    /// Extern statics that are initialized with function pointers to the symbols of the same name.
    fn weak_symbol_extern_statics(
        ecx: &mut MiriInterpCx<'tcx>,
        names: &[&str],
    ) -> InterpResult<'tcx> {
        for name in names {
            assert!(ecx.is_dyn_sym(name), "{name} is not a dynamic symbol");
            let layout = ecx.machine.layouts.const_raw_ptr;
            let ptr = ecx.fn_ptr(FnVal::Other(DynSym::from_str(name)));
            let val = ImmTy::from_scalar(Scalar::from_pointer(ptr, ecx), layout);
            Self::alloc_extern_static(ecx, name, val)?;
        }
        interp_ok(())
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics(ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        if ecx.target_os_is_unix() {
            // "environ" is mandated by POSIX.
            let environ = ecx.machine.env_vars.unix().environ();
            Self::add_extern_static(ecx, "environ", environ);
        }

        match ecx.tcx.sess.target.os.as_ref() {
            "linux" => {
                Self::null_ptr_extern_statics(
                    ecx,
                    &["__cxa_thread_atexit_impl", "__clock_gettime64"],
                )?;
                Self::weak_symbol_extern_statics(ecx, &["getrandom", "gettid", "statx"])?;
            }
            "freebsd" => {
                Self::null_ptr_extern_statics(ecx, &["__cxa_thread_atexit_impl"])?;
            }
            "android" => {
                Self::null_ptr_extern_statics(ecx, &["bsd_signal"])?;
                Self::weak_symbol_extern_statics(ecx, &["signal", "getrandom", "gettid"])?;
            }
            "windows" => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let val = ImmTy::from_int(0, ecx.machine.layouts.u8);
                Self::alloc_extern_static(ecx, "_tls_used", val)?;
            }
            "illumos" | "solaris" => {
                Self::weak_symbol_extern_statics(ecx, &["pthread_setname_np"])?;
            }
            _ => {} // No "extern statics" supported on this target
        }
        interp_ok(())
    }
}
