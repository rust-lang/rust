// This defines the ia32 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options. On ia32 systems
// UEFI systems always run in protected-mode, have the interrupt-controller pre-configured and
// force a single-CPU execution.
// The cdecl ABI is used. It differs from the stdcall or fastcall ABI.
// "i686-unknown-windows" is used to get the minimal subset of windows-specific features.

use crate::spec::Target;

pub fn target() -> Target {
    let mut base = super::uefi_msvc_base::opts();
    base.cpu = "pentium4".to_string();
    base.max_atomic_width = Some(64);

    // We disable MMX and SSE for now, even though UEFI allows using them. Problem is, you have to
    // enable these CPU features explicitly before their first use, otherwise their instructions
    // will trigger an exception. Rust does not inject any code that enables AVX/MMX/SSE
    // instruction sets, so this must be done by the firmware. However, existing firmware is known
    // to leave these uninitialized, thus triggering exceptions if we make use of them. Which is
    // why we avoid them and instead use soft-floats. This is also what GRUB and friends did so
    // far.
    // If you initialize FP units yourself, you can override these flags with custom linker
    // arguments, thus giving you access to full MMX/SSE acceleration.
    base.features = "-mmx,-sse,+soft-float".to_string();

    // Use -GNU here, because of the reason below:
    // Background and Problem:
    //   If we use i686-unknown-windows, the LLVM IA32 MSVC generates compiler intrinsic
    //   _alldiv, _aulldiv, _allrem, _aullrem, _allmul, which will cause undefined symbol.
    //   A real issue is __aulldiv() is referred by __udivdi3() - udivmod_inner!(), from
    //   https://github.com/rust-lang-nursery/compiler-builtins.
    //   As result, rust-lld generates link error finally.
    // Root-cause:
    //   In rust\src\llvm-project\llvm\lib\Target\X86\X86ISelLowering.cpp,
    //   we have below code to use MSVC intrinsics. It assumes MSVC target
    //   will link MSVC library. But that is NOT true in UEFI environment.
    //   UEFI does not link any MSVC or GCC standard library.
    //      if (Subtarget.isTargetKnownWindowsMSVC() ||
    //          Subtarget.isTargetWindowsItanium()) {
    //        // Setup Windows compiler runtime calls.
    //        setLibcallName(RTLIB::SDIV_I64, "_alldiv");
    //        setLibcallName(RTLIB::UDIV_I64, "_aulldiv");
    //        setLibcallName(RTLIB::SREM_I64, "_allrem");
    //        setLibcallName(RTLIB::UREM_I64, "_aullrem");
    //        setLibcallName(RTLIB::MUL_I64, "_allmul");
    //        setLibcallCallingConv(RTLIB::SDIV_I64, CallingConv::X86_StdCall);
    //        setLibcallCallingConv(RTLIB::UDIV_I64, CallingConv::X86_StdCall);
    //        setLibcallCallingConv(RTLIB::SREM_I64, CallingConv::X86_StdCall);
    //        setLibcallCallingConv(RTLIB::UREM_I64, CallingConv::X86_StdCall);
    //        setLibcallCallingConv(RTLIB::MUL_I64, CallingConv::X86_StdCall);
    //      }
    //   The compiler intrisics should be implemented by compiler-builtins.
    //   Unfortunately, compiler-builtins has not provided those intrinsics yet. Such as:
    //      i386/divdi3.S
    //      i386/lshrdi3.S
    //      i386/moddi3.S
    //      i386/muldi3.S
    //      i386/udivdi3.S
    //      i386/umoddi3.S
    // Possible solution:
    //   1. Eliminate Intrinsics generation.
    //      1.1 Choose different target to bypass isTargetKnownWindowsMSVC().
    //      1.2 Remove the "Setup Windows compiler runtime calls" in LLVM
    //   2. Implement Intrinsics.
    //   We evaluated all options.
    //   #2 is hard because we need implement the intrinsics (_aulldiv) generated
    //   from the other intrinscis (__udivdi3) implementation with the same
    //   functionality (udivmod_inner). If we let _aulldiv() call udivmod_inner!(),
    //   then we are in loop. We may have to find another way to implement udivmod_inner!().
    //   #1.2 may break the existing usage.
    //   #1.1 seems the simplest solution today.
    //   The IA32 -gnu calling convention is same as the one defined in UEFI specification.
    //   It uses cdecl, EAX/ECX/EDX as volatile register, and EAX/EDX as return value.
    //   We also checked the LLVM X86TargetLowering, the differences between -gnu and -msvc
    //   is fmodf(f32), longjmp() and TLS. None of them impacts the UEFI code.
    // As a result, we choose -gnu for i686 version before those intrisics are implemented in
    // compiler-builtins. After compiler-builtins implements all required intrinsics, we may
    // remove -gnu and use the default one.
    Target {
        llvm_target: "i686-unknown-windows-gnu".to_string(),
        pointer_width: 32,
        data_layout: "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-f80:32-n8:16:32-a:0:32-S32"
            .to_string(),
        arch: "x86".to_string(),

        options: base,
    }
}
