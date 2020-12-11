# `sanitizer`

The tracking issue for this feature is: [#39699](https://github.com/rust-lang/rust/issues/39699).

------------------------

This feature allows for use of one of following sanitizers:

* [AddressSanitizer][clang-asan] a fast memory error detector.
* [LeakSanitizer][clang-lsan] a run-time memory leak detector.
* [MemorySanitizer][clang-msan] a detector of uninitialized reads.
* [ThreadSanitizer][clang-tsan] a fast data race detector.

To enable a sanitizer compile with `-Zsanitizer=address`, `-Zsanitizer=leak`,
`-Zsanitizer=memory` or `-Zsanitizer=thread`.

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
* `aarch64-fuchsia`
* `aarch64-unknown-linux-gnu`
* `x86_64-apple-darwin`
* `x86_64-fuchsia`
* `x86_64-unknown-freebsd`
* `x86_64-unknown-linux-gnu`

AddressSanitizer works with non-instrumented code although it will impede its
ability to detect some bugs.  It is not expected to produce false positive
reports.

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

# LeakSanitizer

LeakSanitizer is run-time memory leak detector.

LeakSanitizer is supported on the following targets:

* `aarch64-apple-darwin`
* `aarch64-unknown-linux-gnu`
* `x86_64-apple-darwin`
* `x86_64-unknown-linux-gnu`

# MemorySanitizer

MemorySanitizer is detector of uninitialized reads.

MemorySanitizer is supported on the following targets:

* `aarch64-unknown-linux-gnu`
* `x86_64-unknown-freebsd`
* `x86_64-unknown-linux-gnu`

MemorySanitizer requires all program code to be instrumented. C/C++ dependencies
need to be recompiled using Clang with `-fsanitize=memory` option. Failing to
achieve that will result in false positive reports.

## Example

Detecting the use of uninitialized memory. The `-Zbuild-std` flag rebuilds and
instruments the standard library, and is strictly necessary for the correct
operation of the tool. The `-Zsanitizer-track-origins` enables tracking of the
origins of uninitialized memory:

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

[build-std]: https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#build-std

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
* [LeakSanitizer in Clang][clang-lsan]
* [MemorySanitizer in Clang][clang-msan]
* [ThreadSanitizer in Clang][clang-tsan]

[clang-asan]: https://clang.llvm.org/docs/AddressSanitizer.html
[clang-lsan]: https://clang.llvm.org/docs/LeakSanitizer.html
[clang-msan]: https://clang.llvm.org/docs/MemorySanitizer.html
[clang-tsan]: https://clang.llvm.org/docs/ThreadSanitizer.html
