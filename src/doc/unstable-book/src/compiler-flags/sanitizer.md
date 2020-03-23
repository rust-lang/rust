# `sanitizer`

The tracking issue for this feature is: [#39699](https://github.com/rust-lang/rust/issues/39699).

------------------------

This feature allows for use of one of following sanitizers:

* [AddressSanitizer][clang-asan] a faster memory error detector. Can
  detect out-of-bounds access to heap, stack, and globals, use after free, use
  after return, double free, invalid free, memory leaks.
* [LeakSanitizer][clang-lsan] a run-time memory leak detector.
* [MemorySanitizer][clang-msan] a detector of uninitialized reads.
* [ThreadSanitizer][clang-tsan] a fast data race detector.

To enable a sanitizer compile with `-Zsanitizer=...` option, where value is one
of `address`, `leak`, `memory` or `thread`.

# Examples

This sections show various issues that can be detected with sanitizers.  For
simplicity, the examples are prepared under assumption that optimization level
used is zero.

## AddressSanitizer

Stack buffer overflow:

```shell
$ cat a.rs
fn main() {
    let xs = [0, 1, 2, 3];
    let _y = unsafe { *xs.as_ptr().offset(4) };
}
$ rustc -Zsanitizer=address a.rs
$ ./a
=================================================================
==10029==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7ffcc15f43d0 at pc 0x55f77dc015c5 bp 0x7ffcc15f4390 sp 0x7ffcc15f4388
READ of size 4 at 0x7ffcc15f43d0 thread T0
    #0 0x55f77dc015c4 in a::main::hab3bd2a745c2d0ac (/tmp/a+0xa5c4)
    #1 0x55f77dc01cdb in std::rt::lang_start::_$u7b$$u7b$closure$u7d$$u7d$::haa8c76d1faa7b7ca (/tmp/a+0xacdb)
    #2 0x55f77dc90f02 in std::rt::lang_start_internal::_$u7b$$u7b$closure$u7d$$u7d$::hfeb9a1aef9ac820d /rustc/c27f7568bc74c418996892028a629eed5a7f5f00/src/libstd/rt.rs:48:12
    #3 0x55f77dc90f02 in std::panicking::try::do_call::h12f0919717b8e0a6 /rustc/c27f7568bc74c418996892028a629eed5a7f5f00/src/libstd/panicking.rs:288:39
    #4 0x55f77dc926c9 in __rust_maybe_catch_panic /rustc/c27f7568bc74c418996892028a629eed5a7f5f00/src/libpanic_unwind/lib.rs:80:7
    #5 0x55f77dc9197c in std::panicking::try::h413b21cdcd6cfd86 /rustc/c27f7568bc74c418996892028a629eed5a7f5f00/src/libstd/panicking.rs:267:12
    #6 0x55f77dc9197c in std::panic::catch_unwind::hc5cc8ef2fd73424d /rustc/c27f7568bc74c418996892028a629eed5a7f5f00/src/libstd/panic.rs:396:8
    #7 0x55f77dc9197c in std::rt::lang_start_internal::h2039f418ab92218f /rustc/c27f7568bc74c418996892028a629eed5a7f5f00/src/libstd/rt.rs:47:24
    #8 0x55f77dc01c61 in std::rt::lang_start::ha905d28f6b61d691 (/tmp/a+0xac61)
    #9 0x55f77dc0163a in main (/tmp/a+0xa63a)
    #10 0x7f9b3cf5bbba in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x26bba)
    #11 0x55f77dc01289 in _start (/tmp/a+0xa289)

Address 0x7ffcc15f43d0 is located in stack of thread T0 at offset 48 in frame
    #0 0x55f77dc0135f in a::main::hab3bd2a745c2d0ac (/tmp/a+0xa35f)

  This frame has 1 object(s):
    [32, 48) 'xs' <== Memory access at offset 48 overflows this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-buffer-overflow (/tmp/a+0xa5c4) in a::main::hab3bd2a745c2d0ac
Shadow bytes around the buggy address:
  0x1000182b6820: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b6830: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b6840: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b6850: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b6860: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x1000182b6870: 00 00 00 00 f1 f1 f1 f1 00 00[f3]f3 00 00 00 00
  0x1000182b6880: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b6890: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b68a0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b68b0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x1000182b68c0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
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
==10029==ABORTING
```

Use of a stack object after its scope has already ended:

```shell
$ cat b.rs
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
$ rustc -Zsanitizer=address b.rs
$./b
=================================================================
==424427==ERROR: AddressSanitizer: stack-use-after-scope on address 0x7fff67be6be0 at pc 0x5647a3ea4658 bp 0x7fff67be6b90 sp 0x7fff67be6b88
WRITE of size 8 at 0x7fff67be6be0 thread T0
    #0 0x5647a3ea4657 in core::ptr::write_volatile::h4b04601757d0376d (/tmp/b+0xb8657)
    #1 0x5647a3ea4432 in b::main::h5574a756e615c9cf (/tmp/b+0xb8432)
    #2 0x5647a3ea480b in std::rt::lang_start::_$u7b$$u7b$closure$u7d$$u7d$::hd57e7ee01866077e (/tmp/b+0xb880b)
    #3 0x5647a3eab412 in std::panicking::try::do_call::he0421ca82dd11ba3 (.llvm.8083791802951296215) (/tmp/b+0xbf412)
    #4 0x5647a3eacb26 in __rust_maybe_catch_panic (/tmp/b+0xc0b26)
    #5 0x5647a3ea5b66 in std::rt::lang_start_internal::h19bc96b28f670a64 (/tmp/b+0xb9b66)
    #6 0x5647a3ea4788 in std::rt::lang_start::h642d10b4b6965fb8 (/tmp/b+0xb8788)
    #7 0x5647a3ea449a in main (/tmp/b+0xb849a)
    #8 0x7fd1d18b3bba in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x26bba)
    #9 0x5647a3df7299 in _start (/tmp/b+0xb299)

Address 0x7fff67be6be0 is located in stack of thread T0 at offset 32 in frame
    #0 0x5647a3ea433f in b::main::h5574a756e615c9cf (/tmp/b+0xb833f)

  This frame has 1 object(s):
    [32, 40) 'x' <== Memory access at offset 32 is inside this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-use-after-scope (/tmp/b+0xb8657) in core::ptr::write_volatile::h4b04601757d0376d
Shadow bytes around the buggy address:
  0x10006cf74d20: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74d30: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74d40: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74d50: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74d60: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x10006cf74d70: 00 00 00 00 00 00 00 00 f1 f1 f1 f1[f8]f3 f3 f3
  0x10006cf74d80: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74d90: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74da0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74db0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x10006cf74dc0: f1 f1 f1 f1 00 f3 f3 f3 00 00 00 00 00 00 00 00
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
==424427==ABORTING
```

## MemorySanitizer

Use of uninitialized memory. Note that we are using `-Zbuild-std` to instrument
the standard library, and passing `-Zsanitizer-track-origins` to track the
origins of uninitialized memory:

```shell
$ cat src/main.rs
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let a = MaybeUninit::<[usize; 4]>::uninit();
        let a = a.assume_init();
        println!("{}", a[2]);
    }
}

$ export \
  CC=clang \
  CXX=clang++ \
  CFLAGS='-fsanitize=memory -fsanitize-memory-track-origins' \
  CXXFLAGS='-fsanitize=memory -fsanitize-memory-track-origins' \
  RUSTFLAGS='-Zsanitizer=memory -Zsanitizer-memory-track-origins' \
  RUSTDOCFLAGS='-Zsanitizer=memory -Zsanitizer-memory-track-origins'
$ cargo clean
$ cargo -Zbuild-std run --target x86_64-unknown-linux-gnu
==9416==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x560c04f7488a in core::fmt::num::imp::fmt_u64::haa293b0b098501ca $RUST/build/x86_64-unknown-linux-gnu/stage1/lib/rustlib/src/rust/src/libcore/fmt/num.rs:202:16
...
  Uninitialized value was stored to memory at
    #0 0x560c04ae898a in __msan_memcpy.part.0 $RUST/src/llvm-project/compiler-rt/lib/msan/msan_interceptors.cc:1558:3
    #1 0x560c04b2bf88 in memory::main::hd2333c1899d997f5 $CWD/src/main.rs:6:16

  Uninitialized value was created by an allocation of 'a' in the stack frame of function '_ZN6memory4main17hd2333c1899d997f5E'
    #0 0x560c04b2bc50 in memory::main::hd2333c1899d997f5 $CWD/src/main.rs:3
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
