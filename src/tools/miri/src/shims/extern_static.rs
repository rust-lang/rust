//! Provides the `extern static` that this platform expects.

use rustc_span::Symbol;
use rustc_target::spec::Os;

use crate::*;

impl<'tcx> MiriMachine<'tcx> {
    fn add_extern_static(ecx: &mut MiriInterpCx<'tcx>, name: &str, ptr: Pointer) {
        // This got just allocated, so there definitely is a pointer here.
        let ptr = ptr.into_pointer_or_addr().unwrap();
        ecx.machine.extern_statics.try_insert(Symbol::intern(name), ptr).unwrap();
    }

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

    /// Make `ptr` available as a weak symbol with the given name.
    fn add_weak_symbol(
        ecx: &mut MiriInterpCx<'tcx>,
        name: &str,
        ptr: Pointer,
    ) -> InterpResult<'tcx> {
        // Allocate the extra indirection place and add it to the map.
        let layout = ecx.machine.layouts.mut_raw_ptr;
        let place = ecx.allocate(layout, MiriMemoryKind::ExternStatic.into())?;
        ecx.write_scalar(Scalar::from_maybe_pointer(ptr, ecx), &place)?;
        let weak_ptr = place.ptr().into_pointer_or_addr().unwrap();
        ecx.machine.extern_statics_imports.try_insert(Symbol::intern(name), weak_ptr).unwrap();
        interp_ok(())
    }

    /// Extern statics that are initialized with function pointers to the symbols of the same name.
    fn weak_fn_symbols(ecx: &mut MiriInterpCx<'tcx>, names: &[&str]) -> InterpResult<'tcx> {
        for name in names {
            assert!(ecx.is_dyn_sym(name), "{name} is not a dynamic symbol");
            let ptr = ecx.fn_ptr(FnVal::Other(DynSym::from_str(name)));
            Self::add_weak_symbol(ecx, name, ptr.into())?;
        }
        interp_ok(())
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics(ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        if ecx.target_os_is_unix() {
            // "environ" is mandated by POSIX.
            let environ = ecx.machine.env_vars.unix().environ();
            Self::add_extern_static(ecx, "environ", environ);
            // We also provide it as a weak symbol, which is needed on FreeBSD.
            Self::add_weak_symbol(ecx, "environ", environ)?;
        }

        match &ecx.tcx.sess.target.os {
            Os::Linux => {
                Self::weak_fn_symbols(ecx, &["getrandom", "gettid", "statx", "strlen"])?;
            }
            Os::Android => {
                Self::weak_fn_symbols(ecx, &["signal", "getrandom", "gettid", "futimens"])?;
            }
            Os::Windows => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let val = ImmTy::from_int(0, ecx.machine.layouts.u8);
                Self::alloc_extern_static(ecx, "_tls_used", val)?;
            }
            Os::Illumos | Os::Solaris => {
                Self::weak_fn_symbols(ecx, &["pthread_setname_np"])?;
            }
            _ => {} // No "extern statics" supported on this target.
        }

        // Also initialize `missing_weak_symbol`.
        let place = ecx.allocate(ecx.machine.layouts.usize, MiriMemoryKind::ExternStatic.into())?;
        ecx.write_null(&place)?;
        ecx.machine.extern_static_weak_import_default =
            Some(place.ptr().into_pointer_or_addr().unwrap());

        interp_ok(())
    }
}
