//! Implementation of panics backed by libgcc/libunwind (in some form).
//!
//! For background on exception handling and stack unwinding please see
//! "Exception Handling in LLVM" (llvm.org/docs/ExceptionHandling.html) and
//! documents linked from it.
//! These are also good reads:
//!  * <https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html>
//!  * <https://monoinfinito.wordpress.com/series/exception-handling-in-c/>
//!  * <https://www.airs.com/blog/index.php?s=exception+frames>
//!
//! ## A brief summary
//!
//! Exception handling happens in two phases: a search phase and a cleanup
//! phase.
//!
//! In both phases the unwinder walks stack frames from top to bottom using
//! information from the stack frame unwind sections of the current process's
//! modules ("module" here refers to an OS module, i.e., an executable or a
//! dynamic library).
//!
//! For each stack frame, it invokes the associated "personality routine", whose
//! address is also stored in the unwind info section.
//!
//! In the search phase, the job of a personality routine is to examine
//! exception object being thrown, and to decide whether it should be caught at
//! that stack frame. Once the handler frame has been identified, cleanup phase
//! begins.
//!
//! In the cleanup phase, the unwinder invokes each personality routine again.
//! This time it decides which (if any) cleanup code needs to be run for
//! the current stack frame. If so, the control is transferred to a special
//! branch in the function body, the "landing pad", which invokes destructors,
//! frees memory, etc. At the end of the landing pad, control is transferred
//! back to the unwinder and unwinding resumes.
//!
//! Once stack has been unwound down to the handler frame level, unwinding stops
//! and the last personality routine transfers control to the catch block.

use super::dwarf::eh::{self, EHAction, EHContext};
use libc::{c_int, uintptr_t};
use unwind as uw;

// Register ids were lifted from LLVM's TargetLowering::getExceptionPointerRegister()
// and TargetLowering::getExceptionSelectorRegister() for each architecture,
// then mapped to DWARF register numbers via register definition tables
// (typically <arch>RegisterInfo.td, search for "DwarfRegNum").
// See also https://llvm.org/docs/WritingAnLLVMBackend.html#defining-a-register.

#[cfg(target_arch = "x86")]
const UNWIND_DATA_REG: (i32, i32) = (0, 2); // EAX, EDX

#[cfg(target_arch = "x86_64")]
const UNWIND_DATA_REG: (i32, i32) = (0, 1); // RAX, RDX

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const UNWIND_DATA_REG: (i32, i32) = (0, 1); // R0, R1 / X0, X1

#[cfg(target_arch = "m68k")]
const UNWIND_DATA_REG: (i32, i32) = (0, 1); // D0, D1

#[cfg(any(target_arch = "mips", target_arch = "mips64"))]
const UNWIND_DATA_REG: (i32, i32) = (4, 5); // A0, A1

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
const UNWIND_DATA_REG: (i32, i32) = (3, 4); // R3, R4 / X3, X4

#[cfg(target_arch = "s390x")]
const UNWIND_DATA_REG: (i32, i32) = (6, 7); // R6, R7

#[cfg(any(target_arch = "sparc", target_arch = "sparc64"))]
const UNWIND_DATA_REG: (i32, i32) = (24, 25); // I0, I1

#[cfg(target_arch = "hexagon")]
const UNWIND_DATA_REG: (i32, i32) = (0, 1); // R0, R1

#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
const UNWIND_DATA_REG: (i32, i32) = (10, 11); // x10, x11

// The following code is based on GCC's C and C++ personality routines.  For reference, see:
// https://github.com/gcc-mirror/gcc/blob/master/libstdc++-v3/libsupc++/eh_personality.cc
// https://github.com/gcc-mirror/gcc/blob/trunk/libgcc/unwind-c.c

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "arm", not(target_os = "ios"), not(target_os = "watchos"), not(target_os = "netbsd")))] {
        // ARM EHABI personality routine.
        // https://infocenter.arm.com/help/topic/com.arm.doc.ihi0038b/IHI0038B_ehabi.pdf
        //
        // iOS uses the default routine instead since it uses SjLj unwinding.
        #[lang = "eh_personality"]
        unsafe extern "C" fn rust_eh_personality(
            state: uw::_Unwind_State,
            exception_object: *mut uw::_Unwind_Exception,
            context: *mut uw::_Unwind_Context,
        ) -> uw::_Unwind_Reason_Code {
            let state = state as c_int;
            let action = state & uw::_US_ACTION_MASK as c_int;
            let search_phase = if action == uw::_US_VIRTUAL_UNWIND_FRAME as c_int {
                // Backtraces on ARM will call the personality routine with
                // state == _US_VIRTUAL_UNWIND_FRAME | _US_FORCE_UNWIND. In those cases
                // we want to continue unwinding the stack, otherwise all our backtraces
                // would end at __rust_try
                if state & uw::_US_FORCE_UNWIND as c_int != 0 {
                    return continue_unwind(exception_object, context);
                }
                true
            } else if action == uw::_US_UNWIND_FRAME_STARTING as c_int {
                false
            } else if action == uw::_US_UNWIND_FRAME_RESUME as c_int {
                return continue_unwind(exception_object, context);
            } else {
                return uw::_URC_FAILURE;
            };

            // The DWARF unwinder assumes that _Unwind_Context holds things like the function
            // and LSDA pointers, however ARM EHABI places them into the exception object.
            // To preserve signatures of functions like _Unwind_GetLanguageSpecificData(), which
            // take only the context pointer, GCC personality routines stash a pointer to
            // exception_object in the context, using location reserved for ARM's
            // "scratch register" (r12).
            uw::_Unwind_SetGR(context, uw::UNWIND_POINTER_REG, exception_object as uw::_Unwind_Ptr);
            // ...A more principled approach would be to provide the full definition of ARM's
            // _Unwind_Context in our libunwind bindings and fetch the required data from there
            // directly, bypassing DWARF compatibility functions.

            let eh_action = match find_eh_action(context) {
                Ok(action) => action,
                Err(_) => return uw::_URC_FAILURE,
            };
            if search_phase {
                match eh_action {
                    EHAction::None | EHAction::Cleanup(_) => {
                        return continue_unwind(exception_object, context);
                    }
                    EHAction::Catch(_) => {
                        // EHABI requires the personality routine to update the
                        // SP value in the barrier cache of the exception object.
                        (*exception_object).private[5] =
                            uw::_Unwind_GetGR(context, uw::UNWIND_SP_REG);
                        return uw::_URC_HANDLER_FOUND;
                    }
                    EHAction::Terminate => return uw::_URC_FAILURE,
                }
            } else {
                match eh_action {
                    EHAction::None => return continue_unwind(exception_object, context),
                    EHAction::Cleanup(lpad) | EHAction::Catch(lpad) => {
                        uw::_Unwind_SetGR(
                            context,
                            UNWIND_DATA_REG.0,
                            exception_object as uintptr_t,
                        );
                        uw::_Unwind_SetGR(context, UNWIND_DATA_REG.1, 0);
                        uw::_Unwind_SetIP(context, lpad);
                        return uw::_URC_INSTALL_CONTEXT;
                    }
                    EHAction::Terminate => return uw::_URC_FAILURE,
                }
            }

            // On ARM EHABI the personality routine is responsible for actually
            // unwinding a single stack frame before returning (ARM EHABI Sec. 6.1).
            unsafe fn continue_unwind(
                exception_object: *mut uw::_Unwind_Exception,
                context: *mut uw::_Unwind_Context,
            ) -> uw::_Unwind_Reason_Code {
                if __gnu_unwind_frame(exception_object, context) == uw::_URC_NO_REASON {
                    uw::_URC_CONTINUE_UNWIND
                } else {
                    uw::_URC_FAILURE
                }
            }
            // defined in libgcc
            extern "C" {
                fn __gnu_unwind_frame(
                    exception_object: *mut uw::_Unwind_Exception,
                    context: *mut uw::_Unwind_Context,
                ) -> uw::_Unwind_Reason_Code;
            }
        }
    } else {
        // Default personality routine, which is used directly on most targets
        // and indirectly on Windows x86_64 via SEH.
        unsafe extern "C" fn rust_eh_personality_impl(
            version: c_int,
            actions: uw::_Unwind_Action,
            _exception_class: uw::_Unwind_Exception_Class,
            exception_object: *mut uw::_Unwind_Exception,
            context: *mut uw::_Unwind_Context,
        ) -> uw::_Unwind_Reason_Code {
            if version != 1 {
                return uw::_URC_FATAL_PHASE1_ERROR;
            }
            let eh_action = match find_eh_action(context) {
                Ok(action) => action,
                Err(_) => return uw::_URC_FATAL_PHASE1_ERROR,
            };
            if actions as i32 & uw::_UA_SEARCH_PHASE as i32 != 0 {
                match eh_action {
                    EHAction::None | EHAction::Cleanup(_) => uw::_URC_CONTINUE_UNWIND,
                    EHAction::Catch(_) => uw::_URC_HANDLER_FOUND,
                    EHAction::Terminate => uw::_URC_FATAL_PHASE1_ERROR,
                }
            } else {
                match eh_action {
                    EHAction::None => uw::_URC_CONTINUE_UNWIND,
                    EHAction::Cleanup(lpad) | EHAction::Catch(lpad) => {
                        uw::_Unwind_SetGR(
                            context,
                            UNWIND_DATA_REG.0,
                            exception_object as uintptr_t,
                        );
                        uw::_Unwind_SetGR(context, UNWIND_DATA_REG.1, 0);
                        uw::_Unwind_SetIP(context, lpad);
                        uw::_URC_INSTALL_CONTEXT
                    }
                    EHAction::Terminate => uw::_URC_FATAL_PHASE2_ERROR,
                }
            }
        }

        cfg_if::cfg_if! {
            if #[cfg(all(windows, target_arch = "x86_64", target_env = "gnu"))] {
                // On x86_64 MinGW targets, the unwinding mechanism is SEH however the unwind
                // handler data (aka LSDA) uses GCC-compatible encoding.
                #[lang = "eh_personality"]
                #[allow(nonstandard_style)]
                unsafe extern "C" fn rust_eh_personality(
                    exceptionRecord: *mut uw::EXCEPTION_RECORD,
                    establisherFrame: uw::LPVOID,
                    contextRecord: *mut uw::CONTEXT,
                    dispatcherContext: *mut uw::DISPATCHER_CONTEXT,
                ) -> uw::EXCEPTION_DISPOSITION {
                    uw::_GCC_specific_handler(
                        exceptionRecord,
                        establisherFrame,
                        contextRecord,
                        dispatcherContext,
                        rust_eh_personality_impl,
                    )
                }
            } else {
                // The personality routine for most of our targets.
                #[lang = "eh_personality"]
                unsafe extern "C" fn rust_eh_personality(
                    version: c_int,
                    actions: uw::_Unwind_Action,
                    exception_class: uw::_Unwind_Exception_Class,
                    exception_object: *mut uw::_Unwind_Exception,
                    context: *mut uw::_Unwind_Context,
                ) -> uw::_Unwind_Reason_Code {
                    rust_eh_personality_impl(
                        version,
                        actions,
                        exception_class,
                        exception_object,
                        context,
                    )
                }
            }
        }
    }
}

unsafe fn find_eh_action(context: *mut uw::_Unwind_Context) -> Result<EHAction, ()> {
    let lsda = uw::_Unwind_GetLanguageSpecificData(context) as *const u8;
    let mut ip_before_instr: c_int = 0;
    let ip = uw::_Unwind_GetIPInfo(context, &mut ip_before_instr);
    let eh_context = EHContext {
        // The return address points 1 byte past the call instruction,
        // which could be in the next IP range in LSDA range table.
        //
        // `ip = -1` has special meaning, so use wrapping sub to allow for that
        ip: if ip_before_instr != 0 { ip } else { ip.wrapping_sub(1) },
        func_start: uw::_Unwind_GetRegionStart(context),
        get_text_start: &|| uw::_Unwind_GetTextRelBase(context),
        get_data_start: &|| uw::_Unwind_GetDataRelBase(context),
    };
    eh::find_eh_action(lsda, &eh_context)
}
