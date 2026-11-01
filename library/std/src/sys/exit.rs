#[cfg(not(test))]
cfg_select! {
    any(target_family = "unix", target_os = "wasi") => {
        // Used by rustc for checking the definitions of other function with the same symbol names
        //
        // See the `invalid_runtime_symbols_definitions` lint.
        mod runtime_symbols {
            unsafe extern "C" {
                #[rustc_canonical_symbol]
                fn exit(status: core::ffi::c_int) -> !;
            }
        }
    }
    _ => {}
}

pub fn exit(code: i32) -> ! {
    cfg_select! {
        target_os = "hermit" => unsafe { hermit_abi::exit(code) },
        target_os = "motor" => moto_rt::process::exit(code),
        all(target_vendor = "fortanix", target_env = "sgx") => {
            crate::sys::pal::abi::exit_with_code(code as _)
        }
        target_os = "solid_asp3" => {
            rtabort!("exit({}) called", code)
        }
        target_os = "teeos" => {
            let _ = code;
            panic!("TA should not call `exit`")
        }
        target_os = "uefi" => {
            use r_efi::base::Status;

            use crate::os::uefi::env;

            if let (Some(boot_services), Some(handle)) =
                (env::boot_services(), env::try_image_handle())
            {
                let boot_services = boot_services.cast::<r_efi::efi::BootServices>();
                let _ = unsafe {
                    ((*boot_services.as_ptr()).exit)(
                        handle.as_ptr(),
                        Status::from_usize(code as usize),
                        0,
                        crate::ptr::null_mut(),
                    )
                };
            }
            crate::intrinsics::abort_immediate()
        }
        // `exit` causes multiple soundness issues:
        // 1. On glibc, `exit` has been observed to not always be thread-safe and
        //    it is currently unclear whether that is a glibc bug or allowed by
        //    the standard. See #126600 for more.
        // 2. `exit` causes the destructors of C++ thread locals to be run. Since
        //    we use the same mechanisms of destructor registration on platforms
        //    like Linux (`__cxa_thread_atexit_impl`) and macOS (`_tlv_atexit`),
        //    this also applies to our `thread_local!` variables. Unfortunately,
        //    this behaviour can lead to double-frees if the destructor of a TLS
        //    variable being overwritten calls `exit`. See #164243 for more.
        //
        // Thus we use `_exit`, which does not have these problems since it does
        // not perform any cleanup. That's unfortunate, but pretty much
        // unavoidable. A safe `process::exit` function cannot soundly use `exit`
        // while 1. is not resolved.
        any(target_family = "unix", target_os = "wasi") => unsafe {
            libc::_exit(code as crate::ffi::c_int)
        },
        target_os = "vexos" => {
            let _ = code;

            unsafe {
                vex_sdk::vexSystemExitRequest();

                loop {
                    vex_sdk::vexTasksRun();
                }
            }
        }
        target_os = "windows" => unsafe { crate::sys::pal::c::ExitProcess(code as u32) },
        target_os = "xous" => crate::os::xous::ffi::exit(code as u32),
        _ => {
            let _ = code;
            crate::intrinsics::abort_immediate()
        }
    }
}
