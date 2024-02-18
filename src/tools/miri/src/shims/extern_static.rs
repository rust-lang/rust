//! Provides the `extern static` that this platform expects.

use crate::*;

impl<'mir, 'tcx> MiriMachine<'mir, 'tcx> {
    fn alloc_extern_static(
        this: &mut MiriInterpCx<'mir, 'tcx>,
        name: &str,
        val: ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let place = this.allocate(val.layout, MiriMemoryKind::ExternStatic.into())?;
        this.write_immediate(*val, &place)?;
        Self::add_extern_static(this, name, place.ptr());
        Ok(())
    }

    /// Zero-initialized pointer-sized extern statics are pretty common.
    /// Most of them are for weak symbols, which we all set to null (indicating that the
    /// symbol is not supported, and triggering fallback code which ends up calling a
    /// syscall that we do support).
    fn null_ptr_extern_statics(
        this: &mut MiriInterpCx<'mir, 'tcx>,
        names: &[&str],
    ) -> InterpResult<'tcx> {
        for name in names {
            let val = ImmTy::from_int(0, this.machine.layouts.usize);
            Self::alloc_extern_static(this, name, val)?;
        }
        Ok(())
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics(this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
        // "__rust_no_alloc_shim_is_unstable"
        let val = ImmTy::from_int(0, this.machine.layouts.u8);
        Self::alloc_extern_static(this, "__rust_no_alloc_shim_is_unstable", val)?;

        match this.tcx.sess.target.os.as_ref() {
            "linux" => {
                Self::null_ptr_extern_statics(
                    this,
                    &["__cxa_thread_atexit_impl", "getrandom", "statx", "__clock_gettime64"],
                )?;
                // "environ"
                Self::add_extern_static(
                    this,
                    "environ",
                    this.machine.env_vars.environ.as_ref().unwrap().ptr(),
                );
            }
            "freebsd" => {
                Self::null_ptr_extern_statics(this, &["__cxa_thread_atexit_impl"])?;
                // "environ"
                Self::add_extern_static(
                    this,
                    "environ",
                    this.machine.env_vars.environ.as_ref().unwrap().ptr(),
                );
            }
            "android" => {
                Self::null_ptr_extern_statics(this, &["bsd_signal"])?;
                // "signal" -- just needs a non-zero pointer value (function does not even get called),
                // but we arrange for this to call the `signal` function anyway.
                let layout = this.machine.layouts.const_raw_ptr;
                let ptr = this.fn_ptr(FnVal::Other(DynSym::from_str("signal")));
                let val = ImmTy::from_scalar(Scalar::from_pointer(ptr, this), layout);
                Self::alloc_extern_static(this, "signal", val)?;
            }
            "windows" => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let val = ImmTy::from_int(0, this.machine.layouts.u8);
                Self::alloc_extern_static(this, "_tls_used", val)?;
            }
            _ => {} // No "extern statics" supported on this target
        }
        Ok(())
    }
}
