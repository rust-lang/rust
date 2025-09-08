# `sanitizer`

Sanitizers are tools that help detect and prevent various types of bugs and
vulnerabilities in software. They are available in compilers and work by
instrumenting the code to add additional runtime checks. While they provide
powerful tools for identifying bugs or security issues, it's important to note
that using sanitizers can introduce runtime overhead and might not catch all
possible issues. Therefore, they are typically used alongside other best
practices in software development, such as testing and fuzzing, to ensure the
highest level of software quality and security.

The tracking issues for this feature are:

* [#39699](https://github.com/rust-lang/rust/issues/39699).
* [#89653](https://github.com/rust-lang/rust/issues/89653).

------------------------

This feature allows for use of one of following sanitizers:

* Those intended for testing or fuzzing (but not production use):
  * [AddressSanitizer](#addresssanitizer) a fast memory error detector.
  * [HWAddressSanitizer](#hwaddresssanitizer) a memory error detector similar to
    AddressSanitizer, but based on partial hardware assistance.
  * [LeakSanitizer](#leaksanitizer) a run-time memory leak detector.
  * [MemorySanitizer](#memorysanitizer) a detector of uninitialized reads.
  * [ThreadSanitizer](#threadsanitizer) a fast data race detector.

* Those that apart from testing, may be used in production:
  * [ControlFlowIntegrity](#controlflowintegrity) LLVM Control Flow Integrity
    (CFI) provides forward-edge control flow protection.
  * [DataFlowSanitizer](#dataflowsanitizer) a generic dynamic data flow analysis
    framework.
  * [KernelControlFlowIntegrity](#kernelcontrolflowintegrity) LLVM Kernel
    Control Flow Integrity (KCFI) provides forward-edge control flow protection
    for operating systems kernels.
  * [MemTagSanitizer](#memtagsanitizer) fast memory error detector based on
    Armv8.5-A Memory Tagging Extension.
  * [SafeStack](#safestack) provides backward-edge control flow protection by
    separating the stack into safe and unsafe regions.
  * [ShadowCallStack](#shadowcallstack) provides backward-edge control flow
    protection (aarch64 only).

To enable a sanitizer compile with `-Zsanitizer=address`, `-Zsanitizer=cfi`,
`-Zsanitizer=dataflow`,`-Zsanitizer=hwaddress`, `-Zsanitizer=leak`,
`-Zsanitizer=memory`, `-Zsanitizer=memtag`, `-Zsanitizer=shadow-call-stack`, or
`-Zsanitizer=thread`. You might also need the `--target` and `build-std` flags.
If you're working with other languages that are also instrumented with sanitizers,
you might need the `external-clangrt` flag. See the section on
[working with other languages](#working-with-other-languages).

Example:
```shell
$ RUSTFLAGS=-Zsanitizer=address cargo build -Zbuild-std --target x86_64-unknown-linux-gnu
```

Additional options for sanitizers can be passed to LLVM command line argument
processor via LLVM arguments using `llvm-args` codegen option (e.g.,
`-Cllvm-args=-dfsan-combine-pointer-labels-on-load=false`). See the sanitizer
documentation for more information about additional options.

# AddressSanitizer

AddressSanitizer is a memory error detector. It can detect the following types
of bugs:

* Out of bound accesses to heap, stack and globals
* Use after free
* Use after return (runtime flag `ASAN_OPTIONS=detect_stack_use_after_return=1`)
* Use after scope
* Double-free, invalid free
* Memory leaks

The memory leak detection is enabled by default on Linux, and can be enabled
with runtime flag `ASAN_OPTIONS=detect_leaks=1` on macOS.

AddressSanitizer is supported on the following targets:

* `aarch64-apple-darwin`
* `aarch64-unknown-fuchsia`
* `aarch64-unknown-linux-gnu`
* `x86_64-apple-darwin`
* `x86_64-unknown-fuchsia`
* `x86_64-unknown-freebsd`
* `x86_64-unknown-linux-gnu`

AddressSanitizer works with non-instrumented code although it will impede its
ability to detect some bugs.  It is not expected to produce false positive
reports.

See the [Clang AddressSanitizer documentation][clang-asan] for more details.

## Examples

Stack buffer overflow:

```rust
fn main() {
    let xs = [0, 1, 2, 3];
    let _y = unsafe { *xs.as_ptr().offset(4) };
}
```

```shell
$ export RUSTFLAGS=-Zsanitizer=address RUSTDOCFLAGS=-Zsanitizer=address
$ cargo run -Zbuild-std --target x86_64-unknown-linux-gnu
==37882==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7ffe400e6250 at pc 0x5609a841fb20 bp 0x7ffe400e6210 sp 0x7ffe400e6208
READ of size 4 at 0x7ffe400e6250 thread T0
    #0 0x5609a841fb1f in example::main::h628ffc6626ed85b2 /.../src/main.rs:3:23
    ...

Address 0x7ffe400e6250 is located in stack of thread T0 at offset 48 in frame
    #0 0x5609a841f8af in example::main::h628ffc6626ed85b2 /.../src/main.rs:1

  This frame has 1 object(s):
    [32, 48) 'xs' (line 2) <== Memory access at offset 48 overflows this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-buffer-overflow /.../src/main.rs:3:23 in example::main::h628ffc6626ed85b2
Shadow bytes around the buggy address:
  0x100048014bf0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100048014c00: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100048014c10: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100048014c20: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100048014c30: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x100048014c40: 00 00 00 00 f1 f1 f1 f1 00 00[f3]f3 00 00 00 00
  0x100048014c50: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100048014c60: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100048014c70: f1 f1 f1 f1 00 00 f3 f3 00 00 00 00 00 00 00 00
  0x100048014c80: 00 00 00 00 00 00 00 00 00 00 00 00 f1 f1 f1 f1
  0x100048014c90: 00 00 f3 f3 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==37882==ABORTING
```

Use of a stack object after its scope has already ended:

```rust
static mut P: *mut usize = std::ptr::null_mut();

fn main() {
    unsafe {
        {
            let mut x = 0;
            P = &mut x;
        }
        std::ptr::write_volatile(P, 123);
    }
}
```

```shell
$ export RUSTFLAGS=-Zsanitizer=address RUSTDOCFLAGS=-Zsanitizer=address
$ cargo run -Zbuild-std --target x86_64-unknown-linux-gnu
=================================================================
==39249==ERROR: AddressSanitizer: stack-use-after-scope on address 0x7ffc7ed3e1a0 at pc 0x55c98b262a8e bp 0x7ffc7ed3e050 sp 0x7ffc7ed3e048
WRITE of size 8 at 0x7ffc7ed3e1a0 thread T0
    #0 0x55c98b262a8d in core::ptr::write_volatile::he21f1df5a82f329a /.../src/rust/src/libcore/ptr/mod.rs:1048:5
    #1 0x55c98b262cd2 in example::main::h628ffc6626ed85b2 /.../src/main.rs:9:9
    ...

Address 0x7ffc7ed3e1a0 is located in stack of thread T0 at offset 32 in frame
    #0 0x55c98b262bdf in example::main::h628ffc6626ed85b2 /.../src/main.rs:3

  This frame has 1 object(s):
    [32, 40) 'x' (line 6) <== Memory access at offset 32 is inside this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-use-after-scope /.../src/rust/src/libcore/ptr/mod.rs:1048:5 in core::ptr::write_volatile::he21f1df5a82f329a
Shadow bytes around the buggy address:
  0x10000fd9fbe0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10000fd9fbf0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10000fd9fc00: 00 00 00 00 00 00 00 00 00 00 00 00 f1 f1 f1 f1
  0x10000fd9fc10: f8 f8 f3 f3 00 00 00 00 00 00 00 00 00 00 00 00
  0x10000fd9fc20: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x10000fd9fc30: f1 f1 f1 f1[f8]f3 f3 f3 00 00 00 00 00 00 00 00
  0x10000fd9fc40: 00 00 00 00 00 00 00 00 00 00 00 00 f1 f1 f1 f1
  0x10000fd9fc50: 00 00 f3 f3 00 00 00 00 00 00 00 00 00 00 00 00
  0x10000fd9fc60: 00 00 00 00 00 00 00 00 f1 f1 f1 f1 00 00 f3 f3
  0x10000fd9fc70: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10000fd9fc80: 00 00 00 00 f1 f1 f1 f1 00 00 f3 f3 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==39249==ABORTING
```

# ControlFlowIntegrity

The LLVM CFI support in the Rust compiler provides forward-edge control flow
protection for both Rust-compiled code only and for C or C++ and Rust -compiled
code mixed-language binaries, also known as “mixed binaries” (i.e., for when C
or C++ and Rust -compiled code share the same virtual address space), by
aggregating function pointers in groups identified by their return and parameter
types.

LLVM CFI can be enabled with `-Zsanitizer=cfi` and requires LTO (i.e.,
`-Clinker-plugin-lto` or `-Clto`). Cross-language LLVM CFI can be enabled with
`-Zsanitizer=cfi`, and requires the `-Zsanitizer-cfi-normalize-integers` option
to be used with Clang `-fsanitize-cfi-icall-experimental-normalize-integers`
option for cross-language LLVM CFI support, and proper (i.e., non-rustc) LTO
(i.e., `-Clinker-plugin-lto`).

It is recommended to rebuild the standard library with CFI enabled by using the
Cargo build-std feature (i.e., `-Zbuild-std`) when enabling CFI.

See the [Clang ControlFlowIntegrity documentation][clang-cfi] for more details.

## Example 1: Redirecting control flow using an indirect branch/call to an invalid destination

```rust
fn add_one(x: i32) -> i32 {
    x + 1
}

#[unsafe(naked)]
# #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub extern "sysv64" fn add_two(x: i32) {
    // x + 2 preceded by a landing pad/nop block
    std::arch::naked_asm!(
        "
         nop
         nop
         nop
         nop
         nop
         nop
         nop
         nop
         nop
         lea eax, [rdi+2]
         ret
        "
    );
}

fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let answer = do_twice(add_one, 5);

    println!("The answer is: {}", answer);

#   #[cfg(all(target_os = "linux", target_arch = "x86_64"))] {
        println!("With CFI enabled, you should not see the next answer");
        let f: fn(i32) -> i32 = unsafe {
            // Offset 0 is a valid branch/call destination (i.e., the function entry
            // point), but offsets 1-8 within the landing pad/nop block are invalid
            // branch/call destinations (i.e., within the body of the function).
            std::mem::transmute::<*const u8, fn(i32) -> i32>((add_two as *const u8).offset(5))
        };
        let next_answer = do_twice(f, 5);

        println!("The next answer is: {}", next_answer);
#   }
}
```
Fig. 1. Redirecting control flow using an indirect branch/call to an invalid
destination (i.e., within the body of the function).

```shell
$ cargo run --release
   Compiling rust-cfi-1 v0.1.0 (/home/rcvalle/rust-cfi-1)
    Finished release [optimized] target(s) in 0.42s
     Running `target/release/rust-cfi-1`
The answer is: 12
With CFI enabled, you should not see the next answer
The next answer is: 14
$
```
Fig. 2. Build and execution of Fig. 1 with LLVM CFI disabled.

```shell
$ RUSTFLAGS="-Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld -Zsanitizer=cfi" cargo run -Zbuild-std -Zbuild-std-features --release --target x86_64-unknown-linux-gnu
   ...
   Compiling rust-cfi-1 v0.1.0 (/home/rcvalle/rust-cfi-1)
    Finished release [optimized] target(s) in 1m 08s
     Running `target/x86_64-unknown-linux-gnu/release/rust-cfi-1`
The answer is: 12
With CFI enabled, you should not see the next answer
Illegal instruction
$
```
Fig. 3. Build and execution of Fig. 1 with LLVM CFI enabled.

When LLVM CFI is enabled, if there are any attempts to change/hijack control
flow using an indirect branch/call to an invalid destination, the execution is
terminated (see Fig. 3).

## Example 2: Redirecting control flow using an indirect branch/call to a function with a different number of parameters

```rust
use std::mem;

fn add_one(x: i32) -> i32 {
    x + 1
}

fn add_two(x: i32, _y: i32) -> i32 {
    x + 2
}

fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let answer = do_twice(add_one, 5);

    println!("The answer is: {}", answer);

    println!("With CFI enabled, you should not see the next answer");
    let f: fn(i32) -> i32 =
        unsafe { mem::transmute::<*const u8, fn(i32) -> i32>(add_two as *const u8) };
    let next_answer = do_twice(f, 5);

    println!("The next answer is: {}", next_answer);
}
```
Fig. 4. Redirecting control flow using an indirect branch/call to a function
with a different number of parameters than arguments intended/passed in the
call/branch site.

```shell
$ cargo run --release
   Compiling rust-cfi-2 v0.1.0 (/home/rcvalle/rust-cfi-2)
    Finished release [optimized] target(s) in 0.43s
     Running `target/release/rust-cfi-2`
The answer is: 12
With CFI enabled, you should not see the next answer
The next answer is: 14
$
```
Fig. 5. Build and execution of Fig. 4 with LLVM CFI disabled.

```shell
$ RUSTFLAGS="-Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld -Zsanitizer=cfi" cargo run -Zbuild-std -Zbuild-std-features --release --target x86_64-unknown-linux-gnu
   ...
   Compiling rust-cfi-2 v0.1.0 (/home/rcvalle/rust-cfi-2)
    Finished release [optimized] target(s) in 1m 08s
     Running `target/x86_64-unknown-linux-gnu/release/rust-cfi-2`
The answer is: 12
With CFI enabled, you should not see the next answer
Illegal instruction
$
```
Fig. 6. Build and execution of Fig. 4 with LLVM CFI enabled.

When LLVM CFI is enabled, if there are any attempts to change/hijack control
flow using an indirect branch/call to a function with different number of
parameters than arguments intended/passed in the call/branch site, the
execution is also terminated (see Fig. 6).

## Example 3: Redirecting control flow using an indirect branch/call to a function with different return and parameter types

```rust
use std::mem;

fn add_one(x: i32) -> i32 {
    x + 1
}

fn add_two(x: i64) -> i64 {
    x + 2
}

fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let answer = do_twice(add_one, 5);

    println!("The answer is: {}", answer);

    println!("With CFI enabled, you should not see the next answer");
    let f: fn(i32) -> i32 =
        unsafe { mem::transmute::<*const u8, fn(i32) -> i32>(add_two as *const u8) };
    let next_answer = do_twice(f, 5);

    println!("The next answer is: {}", next_answer);
}
```
Fig. 7. Redirecting control flow using an indirect branch/call to a function
with different return and parameter types than the return type expected and
arguments intended/passed at the call/branch site.

```shell
$ cargo run --release
   Compiling rust-cfi-3 v0.1.0 (/home/rcvalle/rust-cfi-3)
    Finished release [optimized] target(s) in 0.44s
     Running `target/release/rust-cfi-3`
The answer is: 12
With CFI enabled, you should not see the next answer
The next answer is: 14
$
```
Fig. 8. Build and execution of Fig. 7 with LLVM CFI disabled.

```shell
$ RUSTFLAGS="-Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld -Zsanitizer=cfi" cargo run -Zbuild-std -Zbuild-std-features --release --target x86_64-unknown-linux-gnu
   ...
   Compiling rust-cfi-3 v0.1.0 (/home/rcvalle/rust-cfi-3)
    Finished release [optimized] target(s) in 1m 07s
     Running `target/x86_64-unknown-linux-gnu/release/rust-cfi-3`
The answer is: 12
With CFI enabled, you should not see the next answer
Illegal instruction
$
```
Fig. 9. Build and execution of Fig. 7 with LLVM CFI enabled.

When LLVM CFI is enabled, if there are any attempts to change/hijack control
flow using an indirect branch/call to a function with different return and
parameter types than the return type expected and arguments intended/passed in
the call/branch site, the execution is also terminated (see Fig. 9).

## Example 4: Redirecting control flow using an indirect branch/call to a function with different return and parameter types across the FFI boundary

```ignore (cannot-test-this-because-uses-custom-build)
int
do_twice(int (*fn)(int), int arg)
{
    return fn(arg) + fn(arg);
}
```
Fig. 10. Example C library.

```ignore (cannot-test-this-because-uses-custom-build)
use std::mem;

#[link(name = "foo")]
extern "C" {
    fn do_twice(f: unsafe extern "C" fn(i32) -> i32, arg: i32) -> i32;
}

unsafe extern "C" fn add_one(x: i32) -> i32 {
    x + 1
}

unsafe extern "C" fn add_two(x: i64) -> i64 {
    x + 2
}

fn main() {
    let answer = unsafe { do_twice(add_one, 5) };

    println!("The answer is: {}", answer);

    println!("With CFI enabled, you should not see the next answer");
    let f: unsafe extern "C" fn(i32) -> i32 = unsafe {
        mem::transmute::<*const u8, unsafe extern "C" fn(i32) -> i32>(add_two as *const u8)
    };
    let next_answer = unsafe { do_twice(f, 5) };

    println!("The next answer is: {}", next_answer);
}
```
Fig. 11. Redirecting control flow using an indirect branch/call to a function
with different return and parameter types than the return type expected and
arguments intended/passed in the call/branch site, across the FFI boundary.

```shell
$ make
mkdir -p target/release
clang -I. -Isrc -Wall -c src/foo.c -o target/release/libfoo.o
llvm-ar rcs target/release/libfoo.a target/release/libfoo.o
RUSTFLAGS="-L./target/release -Clinker=clang -Clink-arg=-fuse-ld=lld" cargo build --release
   Compiling rust-cfi-4 v0.1.0 (/home/rcvalle/rust-cfi-4)
    Finished release [optimized] target(s) in 0.49s
$ ./target/release/rust-cfi-4
The answer is: 12
With CFI enabled, you should not see the next answer
The next answer is: 14
$
```
Fig. 12. Build and execution of Figs. 10–11 with LLVM CFI disabled.

```shell
$ make
mkdir -p target/release
clang -I. -Isrc -Wall -flto -fsanitize=cfi -fsanitize-cfi-icall-experimental-normalize-integers -fvisibility=hidden -c -emit-llvm src/foo.c -o target/release/libfoo.bc
llvm-ar rcs target/release/libfoo.a target/release/libfoo.bc
RUSTFLAGS="-L./target/release -Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld -Zsanitizer=cfi -Zsanitizer-cfi-normalize-integers" cargo build -Zbuild-std -Zbuild-std-features --release --target x86_64-unknown-linux-gnu
   ...
   Compiling rust-cfi-4 v0.1.0 (/home/rcvalle/rust-cfi-4)
    Finished release [optimized] target(s) in 1m 06s
$ ./target/x86_64-unknown-linux-gnu/release/rust-cfi-4
The answer is: 12
With CFI enabled, you should not see the next answer
Illegal instruction
$
```
Fig. 13. Build and execution of FIgs. 10–11 with LLVM CFI enabled.

When LLVM CFI is enabled, if there are any attempts to redirect control flow
using an indirect branch/call to a function with different return and parameter
types than the return type expected and arguments intended/passed in the
call/branch site, even across the FFI boundary and for extern "C" function types
indirectly called (i.e., callbacks/function pointers) across the FFI boundary,
the execution is also terminated (see Fig. 13).

# HWAddressSanitizer

HWAddressSanitizer is a newer variant of AddressSanitizer that consumes much
less memory.

HWAddressSanitizer is supported on the following targets:

* `aarch64-linux-android`
* `aarch64-unknown-linux-gnu`

HWAddressSanitizer requires `tagged-globals` target feature to instrument
globals. To enable this target feature compile with `-C
target-feature=+tagged-globals`

See the [Clang HWAddressSanitizer documentation][clang-hwasan] for more details.

## Example

Heap buffer overflow:

```rust
fn main() {
    let xs = vec![0, 1, 2, 3];
    let _y = unsafe { *xs.as_ptr().offset(4) };
}
```

```shell
$ rustc main.rs -Zsanitizer=hwaddress -C target-feature=+tagged-globals -C
linker=aarch64-linux-gnu-gcc -C link-arg=-fuse-ld=lld --target
aarch64-unknown-linux-gnu
```

```shell
$ ./main
==241==ERROR: HWAddressSanitizer: tag-mismatch on address 0xefdeffff0050 at pc 0xaaaae0ae4a98
READ of size 4 at 0xefdeffff0050 tags: 2c/00 (ptr/mem) in thread T0
    #0 0xaaaae0ae4a94  (/.../main+0x54a94)
    ...

[0xefdeffff0040,0xefdeffff0060) is a small allocated heap chunk; size: 32 offset: 16
0xefdeffff0050 is located 0 bytes to the right of 16-byte region [0xefdeffff0040,0xefdeffff0050)
allocated here:
    #0 0xaaaae0acb80c  (/.../main+0x3b80c)
    ...

Thread: T0 0xeffe00002000 stack: [0xffffc28ad000,0xffffc30ad000) sz: 8388608 tls: [0xffffaa10a020,0xffffaa10a7d0)
Memory tags around the buggy address (one tag corresponds to 16 bytes):
  0xfefcefffef80: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffef90: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffefa0: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffefb0: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffefc0: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffefd0: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffefe0: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefcefffeff0: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
=>0xfefceffff000: d7  d7  05  00  2c [00] 00  00  00  00  00  00  00  00  00  00
  0xfefceffff010: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff020: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff030: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff040: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff050: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff060: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff070: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
  0xfefceffff080: 00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00
Tags for short granules around the buggy address (one tag corresponds to 16 bytes):
  0xfefcefffeff0: ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
=>0xfefceffff000: ..  ..  8c  ..  .. [..] ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
  0xfefceffff010: ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
See https://clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html#short-granules for a description of short granule tags
Registers where the failure occurred (pc 0xaaaae0ae4a98):
    x0  2c00efdeffff0050  x1  0000000000000004  x2  0000000000000004  x3  0000000000000000
    x4  0000fffefc30ac37  x5  000000000000005d  x6  00000ffffc30ac37  x7  0000efff00000000
    x8  2c00efdeffff0050  x9  0200efff00000000  x10 0000000000000000  x11 0200efff00000000
    x12 0200effe00000310  x13 0200effe00000310  x14 0000000000000008  x15 5d00ffffc30ac360
    x16 0000aaaae0ad062c  x17 0000000000000003  x18 0000000000000001  x19 0000ffffc30ac658
    x20 4e00ffffc30ac6e0  x21 0000aaaae0ac5e10  x22 0000000000000000  x23 0000000000000000
    x24 0000000000000000  x25 0000000000000000  x26 0000000000000000  x27 0000000000000000
    x28 0000000000000000  x29 0000ffffc30ac5a0  x30 0000aaaae0ae4a98
SUMMARY: HWAddressSanitizer: tag-mismatch (/.../main+0x54a94)
```

# KernelControlFlowIntegrity

The LLVM Kernel Control Flow Integrity (CFI) support to the Rust compiler
initially provides forward-edge control flow protection for operating systems
kernels for Rust-compiled code only by aggregating function pointers in groups
identified by their return and parameter types. (See [LLVM commit cff5bef "KCFI
sanitizer"](https://github.com/llvm/llvm-project/commit/cff5bef948c91e4919de8a5fb9765e0edc13f3de).)

Forward-edge control flow protection for C or C++ and Rust -compiled code "mixed
binaries" (i.e., for when C or C++ and Rust -compiled code share the same
virtual address space) will be provided in later work by defining and using
compatible type identifiers (see Type metadata in the design document in the
tracking issue [#89653](https://github.com/rust-lang/rust/issues/89653)).

LLVM KCFI can be enabled with `-Zsanitizer=kcfi`.

LLVM KCFI is supported on the following targets:

* `aarch64-linux-android`
* `aarch64-unknown-linux-gnu`
* `x86_64-linux-android`
* `x86_64-unknown-linux-gnu`

See the [Clang KernelControlFlowIntegrity documentation][clang-kcfi] for more
details.

# DataFlowSanitizer

DataFlowSanitizer is a generalised dynamic data flow analysis.

Unlike other Sanitizer tools, this tool is not designed to detect a specific
class of bugs on its own. Instead, it provides a generic dynamic data flow
analysis framework to be used by clients to help detect application-specific
issues within their own code.

DataFlowSanitizer is supported on the following targets:

* `x86_64-unknown-linux-gnu`

See the [Clang DataFlowSanitizer documentation][clang-dataflow] for more details.

# KernelAddressSanitizer

KernelAddressSanitizer (KASAN) is a freestanding version of AddressSanitizer
which is suitable for detecting memory errors in programs which do not have a
runtime environment, such as operating system kernels. KernelAddressSanitizer
requires manual implementation of the underlying functions used for tracking
KernelAddressSanitizer state.

KernelAddressSanitizer is supported on the following targets:

* `aarch64-unknown-none`
* `riscv64gc-unknown-none-elf`
* `riscv64imac-unknown-none-elf`
* `x86_64-unknown-none`

See the [Linux Kernel's KernelAddressSanitizer documentation][linux-kasan] for
more details.

# LeakSanitizer

LeakSanitizer is run-time memory leak detector.

LeakSanitizer is supported on the following targets:

* `aarch64-unknown-linux-gnu`
* `x86_64-apple-darwin`
* `x86_64-unknown-linux-gnu`

See the [Clang LeakSanitizer documentation][clang-lsan] for more details.

# MemorySanitizer

MemorySanitizer is detector of uninitialized reads.

MemorySanitizer is supported on the following targets:

* `aarch64-unknown-linux-gnu`
* `x86_64-unknown-freebsd`
* `x86_64-unknown-linux-gnu`

MemorySanitizer requires all program code to be instrumented. C/C++ dependencies
need to be recompiled using Clang with `-fsanitize=memory` option. Failing to
achieve that will result in false positive reports.

See the [Clang MemorySanitizer documentation][clang-msan] for more details.

## Example

Detecting the use of uninitialized memory. The `-Zbuild-std` flag rebuilds and
instruments the standard library, and is strictly necessary for the correct
operation of the tool. The `-Zsanitizer-memory-track-origins` enables tracking
of the origins of uninitialized memory:

```rust
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let a = MaybeUninit::<[usize; 4]>::uninit();
        let a = a.assume_init();
        println!("{}", a[2]);
    }
}
```

```shell
$ export \
  RUSTFLAGS='-Zsanitizer=memory -Zsanitizer-memory-track-origins' \
  RUSTDOCFLAGS='-Zsanitizer=memory -Zsanitizer-memory-track-origins'
$ cargo clean
$ cargo run -Zbuild-std --target x86_64-unknown-linux-gnu
==9416==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x560c04f7488a in core::fmt::num::imp::fmt_u64::haa293b0b098501ca $RUST/build/x86_64-unknown-linux-gnu/stage1/lib/rustlib/src/rust/src/libcore/fmt/num.rs:202:16
...
  Uninitialized value was stored to memory at
    #0 0x560c04ae898a in __msan_memcpy.part.0 $RUST/src/llvm-project/compiler-rt/lib/msan/msan_interceptors.cc:1558:3
    #1 0x560c04b2bf88 in memory::main::hd2333c1899d997f5 $CWD/src/main.rs:6:16

  Uninitialized value was created by an allocation of 'a' in the stack frame of function '_ZN6memory4main17hd2333c1899d997f5E'
    #0 0x560c04b2bc50 in memory::main::hd2333c1899d997f5 $CWD/src/main.rs:3
```

# MemTagSanitizer

MemTagSanitizer detects a similar class of errors as AddressSanitizer and HardwareAddressSanitizer, but with lower overhead suitable for use as hardening for production binaries.

MemTagSanitizer is supported on the following targets:

* `aarch64-linux-android`
* `aarch64-unknown-linux-gnu`

MemTagSanitizer requires hardware support and the `mte` target feature.
To enable this target feature compile with `-C target-feature="+mte"`.

See the [LLVM MemTagSanitizer documentation][llvm-memtag] for more details.

# SafeStack

SafeStack provides backward edge control flow protection by separating the stack into data which is only accessed safely (the safe stack) and all other data (the unsafe stack).

SafeStack can be enabled with the `-Zsanitizer=safestack` option and is supported on the following targets:

* `x86_64-unknown-linux-gnu`

See the [Clang SafeStack documentation][clang-safestack] for more details.

# ShadowCallStack

ShadowCallStack provides backward edge control flow protection by storing a function's return address in a separately allocated 'shadow call stack'
and loading the return address from that shadow call stack.
AArch64 and RISC-V both have a platform register defined in their ABIs, which is `x18` and `x3`/`gp` respectively, that can optionally be reserved for this purpose.
Software support from the operating system and runtime may be required depending on the target platform which is detailed in the remaining section.
See the [Clang ShadowCallStack documentation][clang-scs] for more details.

ShadowCallStack can be enabled with `-Zsanitizer=shadow-call-stack` option and is supported on the following targets:

## AArch64 family

ShadowCallStack requires the use of the ABI defined platform register, `x18`, which is required for code generation purposes.
When `x18` is not reserved, and is instead used as a scratch register subsequently, enabling ShadowCallStack would lead to undefined behaviour
due to corruption of return address or invalid memory access when the instrumentation restores return register to the link register `lr` from the
already clobbered `x18` register.
In other words, code that is calling into or called by functions instrumented with ShadowCallStack must reserve the `x18` register or preserve its value.

### `aarch64-linux-android` and `aarch64-unknown-fuchsia`/`aarch64-fuchsia`

This target already reserves the `x18` register.
A runtime must be provided by the application or operating system.
If `bionic` is used on this target, the software support is provided.
Otherwise, a runtime needs to prepare a memory region and points `x18` to the region which serves as the shadow call stack.

### `aarch64-unknown-none`

In addition to support from a runtime by the application or operating system, the `-Zfixed-x18` flag is also mandatory.

## RISC-V 64 family

ShadowCallStack uses either the `gp` register for software shadow stack, also known as `x3`, or the `ssp` register if [`Zicfiss`][riscv-zicfiss] extension is available.
`gp`/`x3` is currently always reserved and available for ShadowCallStack instrumentation, and `ssp` in case of `Zicfiss` is only accessible through its dedicated shadow stack instructions.

Support from the runtime and operating system is required when `gp`/`x3` is used for software shadow stack.
A runtime must prepare a memory region and point `gp`/`x3` to the region before executing the code.

The following targets support ShadowCallStack.

* `riscv64imac-unknown-none-elf`
* `riscv64gc-unknown-none-elf`
* `riscv64gc-unknown-fuchsia`

# ThreadSanitizer

ThreadSanitizer is a data race detection tool. It is supported on the following
targets:

* `aarch64-apple-darwin`
* `aarch64-unknown-linux-gnu`
* `x86_64-apple-darwin`
* `x86_64-unknown-freebsd`
* `x86_64-unknown-linux-gnu`

To work correctly ThreadSanitizer needs to be "aware" of all synchronization
operations in a program. It generally achieves that through combination of
library interception (for example synchronization performed through
`pthread_mutex_lock` / `pthread_mutex_unlock`) and compile time instrumentation
(e.g. atomic operations). Using it without instrumenting all the program code
can lead to false positive reports.

ThreadSanitizer does not support atomic fences `std::sync::atomic::fence`,
nor synchronization performed using inline assembly code.

See the [Clang ThreadSanitizer documentation][clang-tsan] for more details.

## Example

```rust
static mut A: usize = 0;

fn main() {
    let t = std::thread::spawn(|| {
        unsafe { A += 1 };
    });
    unsafe { A += 1 };

    t.join().unwrap();
}
```

```shell
$ export RUSTFLAGS=-Zsanitizer=thread RUSTDOCFLAGS=-Zsanitizer=thread
$ cargo run -Zbuild-std --target x86_64-unknown-linux-gnu
==================
WARNING: ThreadSanitizer: data race (pid=10574)
  Read of size 8 at 0x5632dfe3d030 by thread T1:
    #0 example::main::_$u7b$$u7b$closure$u7d$$u7d$::h23f64b0b2f8c9484 ../src/main.rs:5:18 (example+0x86cec)
    ...

  Previous write of size 8 at 0x5632dfe3d030 by main thread:
    #0 example::main::h628ffc6626ed85b2 /.../src/main.rs:7:14 (example+0x868c8)
    ...
    #11 main <null> (example+0x86a1a)

  Location is global 'example::A::h43ac149ddf992709' of size 8 at 0x5632dfe3d030 (example+0x000000bd9030)
```

# Instrumentation of external dependencies and std

The sanitizers to varying degrees work correctly with partially instrumented
code. On the one extreme is LeakSanitizer that doesn't use any compile time
instrumentation, on the other is MemorySanitizer that requires that all program
code to be instrumented (failing to achieve that will inevitably result in
false positives).

It is strongly recommended to combine sanitizers with recompiled and
instrumented standard library, for example using [cargo `-Zbuild-std`
functionality][build-std].

[build-std]: ../../cargo/reference/unstable.html#build-std

# Working with other languages

Sanitizers rely on compiler runtime libraries to function properly. Rust links
in its own compiler runtime which might conflict with runtimes required by
languages such as C++. Since Rust's runtime doesn't always contain the symbols
required by C++ instrumented code, you might need to skip linking it so another
runtime can be linked instead.

A separate unstable option `-Zexternal-clangrt` can be used to make rustc skip
linking the compiler runtime for the sanitizer. This will require you to link
in an external runtime, such as from clang instead.

# Build scripts and procedural macros

Use of sanitizers together with build scripts and procedural macros is
technically possible, but in almost all cases it would be best avoided.  This
is especially true for procedural macros which would require an instrumented
version of rustc.

In more practical terms when using cargo always remember to pass `--target`
flag, so that rustflags will not be applied to build scripts and procedural
macros.

# Symbolizing the Reports

Sanitizers produce symbolized stacktraces when llvm-symbolizer binary is in `PATH`.

# Additional Information

* [Sanitizers project page](https://github.com/google/sanitizers/wiki/)
* [AddressSanitizer in Clang][clang-asan]
* [ControlFlowIntegrity in Clang][clang-cfi]
* [DataFlowSanitizer in Clang][clang-dataflow]
* [HWAddressSanitizer in Clang][clang-hwasan]
* [Linux Kernel's KernelAddressSanitizer documentation][linux-kasan]
* [LeakSanitizer in Clang][clang-lsan]
* [MemorySanitizer in Clang][clang-msan]
* [MemTagSanitizer in LLVM][llvm-memtag]
* [ThreadSanitizer in Clang][clang-tsan]

[clang-asan]: https://clang.llvm.org/docs/AddressSanitizer.html
[clang-cfi]: https://clang.llvm.org/docs/ControlFlowIntegrity.html
[clang-dataflow]: https://clang.llvm.org/docs/DataFlowSanitizer.html
[clang-hwasan]: https://clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html
[clang-kcfi]: https://clang.llvm.org/docs/ControlFlowIntegrity.html#fsanitize-kcfi
[clang-lsan]: https://clang.llvm.org/docs/LeakSanitizer.html
[clang-msan]: https://clang.llvm.org/docs/MemorySanitizer.html
[clang-safestack]: https://clang.llvm.org/docs/SafeStack.html
[clang-scs]: https://clang.llvm.org/docs/ShadowCallStack.html
[clang-tsan]: https://clang.llvm.org/docs/ThreadSanitizer.html
[linux-kasan]: https://www.kernel.org/doc/html/latest/dev-tools/kasan.html
[llvm-memtag]: https://llvm.org/docs/MemTagSanitizer.html
[riscv-zicfiss]: https://github.com/riscv/riscv-cfi/blob/3f8e450c481ac303bd5643444f7a89672f24476e/src/cfi_backward.adoc
