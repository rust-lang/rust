# `instrument-mcount`

Insert calls to a counting function at the entry of each function. Traditionally, the name of this function was
mcount, but the exact name may vary depending on target and option usage.

The counting function is a special function which does not typically follow a target's ABI. It generally takes
two arguments, the address of the calling function and the address of the called function. It was intially used
to profile applications, but has expanded to other usages (for example [ftrace on Linux](https://docs.kernel.org/trace/ftrace.html)).

Supported options:

 - `no`, `n`, `off`: Do no enable instrumentation. The default option. This requires, and enables frame pointer generation.
 - `yes`, `y`, `on`: Enable mcount based function instrumentation.
 - `fentry`: Enable fentry based function instrument, where supported. The calling conventions for this are different than mcount, with less overhead, and no frame pointer requirements. This counting function is always named `__fentry__`. This is only available on x86 and s390x targets.

|target                   |mcount function|supports fentry|ABI notes|
|---                      |---            |---            |---      |
|aarch64-apple-darwin     | `\u{1}mcount` |               |         |
|aarch64-pc-windows-msvc  | `mcount`      |               |         |
|aarch64-unknown-linux-gnu| `_mcount`     |               |         |
|i686-pc-windows-msvc     | `mcount`      |              x|         |
|i686-unknown-linux-gnu   | `mcount`      |              x|         |
|x86_64-pc-windows-gnu    | `_mcount`     |              x|         |
|x86_64-pc-windows-msvc   | `mcount`      |              x|         |
|x86_64-unknown-linux-gnu | `mcount`      |              x|        1|

On arm eabi targets, the mcount function is usually named `__gnu_mcount_nc`, though some targets may use different names. Implementers of counting function should consult the target specific documentation for quirks of each ABI function.

1. On x86-64, mcount and fentry must preserve the argument registers `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`, `r9`. When using fentry, the stack pointer `rsp` may need aligned to meet ABI requirements.

## Implementing custom counting functions

In essence, this is implementing the function `fn mcount(caller: *const std::ffi::c_void, callee: *const std::ffi::c_void)`. The calling convention for mcount follows its own ABI, which isn't usually the standard ABI for the target, but is enforced by preexisting convention.

A trivial example on x86_64-unknown-linux-gnu looks something like the following. The `#[instrument_fn]` attribute can be used to disable profiling to simplify writing counting functions, but implementors must be very careful when calling other functions (or closures which fail to inline) as they may also call mcount.

The following example can be compiled with `-Zinstrument-mcount=yes` or `-Zinstrument-mcount=fentry` on an x86_64-unknown-linux-gnu target. It is also acceptable to link objects with different usages of `-Zinstrument-mcount`, however doing so will require implementing both `__fentry__` and `mcount` on targets which support both.

```rust
#![feature(instrument_fn)]
#![feature(abi_custom)]

fn main() {
    // Ensure all the early startup occurs before attempting to call this trivial, single-threaded
    // counting function.
    unsafe {
        PROFILING_ENABLED = true;
    }
    println!("main() called");
    unsafe {
        PROFILING_ENABLED = false;
    }
}

// This example is not threadsafe.
pub static mut IN_MCOUNT: isize = 0;
pub static mut PROFILING_ENABLED: bool = false;

#[unsafe(no_mangle)]
#[instrument_fn = "off"]
unsafe extern "C" fn __count_fn(caller: u64, callee: u64) {
    unsafe {
        if IN_MCOUNT == 0 && PROFILING_ENABLED {
            IN_MCOUNT += 1;
            {
                println!("mcount: call from 0x{caller:x} to 0x{callee:x}");
            }
            IN_MCOUNT -= 1;
        }
    }
}

// Define a custom mcount function. This may partially or fully override the glibc
// implementation depending on linker options.
#[unsafe(naked)]
#[unsafe(no_mangle)]
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe extern "custom" fn mcount() {
    core::arch::naked_asm!(
        // A simplified version based on the glibc x86-64 mcount wrapper.
        // Save register arguments to the stack, and call the mcount above.
        "push rax",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push r8",
        "push r9",
        "mov rsi, 56[rsp]",
        "mov rdi, 8[rbp]",
        "call __count_fn",
        "pop r9",
        "pop r8",
        "pop rdi",
        "pop rsi",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "ret",
    )
}

// Supply a custom __fentry__ instead of glibc's. This has the same linker
// restrictions as noted with mcount, but does not require a frame pointer.
#[unsafe(naked)]
#[unsafe(no_mangle)]
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe extern "custom" fn __fentry__() {
    core::arch::naked_asm!(
        // __fentry__ is called before any other prologue actions, be careful
        // with stack alignment. The stack slots look something like:
        // [...]
        // [caller return address]
        // [callee return address] <- top of stack
        "sub rsp, 8",
        "push rax",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push r8",
        "push r9",
        "mov rsi, 64[rsp]",
        "mov rdi, 72[rsp]",
        "call __count_fn",
        "pop r9",
        "pop r8",
        "pop rdi",
        "pop rsi",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "add rsp, 8",
        "ret",
    )
}
```

When run, the above program should produce output similar to:
```txt
mcount: call from 0x5614c97d778a to 0x5614c97d76e5
main() called
```
