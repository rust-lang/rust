# Sanitizers

## Introduction

Sanitizers are a set of compiler-based runtime error detection tools that
instrument programs to detect bugs during execution. They work by instrumenting
code at compile time and runtime to monitor program behavior and detect specific
classes of errors at runtime. Sanitizers enable precise, low-overhead runtime
bug detection, improving software reliability and security.

This option allows for use of one or more of these sanitizers:

  * [AddressSanitizer (ASan)](#addresssanitizer): Detects memory errors (e.g.,
    buffer overflows, use after free).
  * [LeakSanitizer (LSan)](#leaksanitizer): Detects memory leaks either as part
    of AddressSanitizer or as a standalone tool.

These are the valid values for this option for targets that support one or more
of these sanitizers:

| Target                      | Sanitizers      |
|-----------------------------|-----------------|
| aarch64-apple-darwin        | address         |
| aarch64-unknown-linux-gnu   | address, leak   |
| i686-pc-windows-msvc        | address         |
| i686-unknown-linux-gnu      | address         |
| x86_64-apple-darwin         | address, leak   |
| x86_64-pc-windows-msvc      | address         |
| x86_64-unknown-linux-gnu    | address, leak   |

## AddressSanitizer

AddressSanitizer (ASan) detects memory errors by instrumenting code at compile
time and runtime to mark regions around allocated memory (i.e., red zones) as
unaddressable (i.e., poisoned), quarantine and mark deallocated memory as
unaddressable, and add checks before memory accesses. It uses a shadow memory
mapping to store metadata information about whether a memory region is
addressable. It can detect:

* Heap-based buffer overflows, Stack-based buffer overflows, and other variants
  of out-of-bounds reads and writes.
* Use after free, double free, and other variants of expired pointer dereference
  (also known as “dangling pointer”).
* Initialization order bugs (such as [“Static Initialization Order
  Fiasco”](https://en.cppreference.com/w/cpp/language/siof)).
* Memory leaks.

AddressSanitizer uses both instrumentation at compile time and runtime. It is
recommended to recompile all code using AddressSanitizer for best results. If
parts of the compiled code are not instrumented, AddressSanitizer may not detect
certain memory errors or detect false positives.

AddressSanitizer increases memory usage and also impacts performance due to the
red zones and shadow memory mapping, and the added checks before memory
accesses. AddressSanitizer and its runtime are not suitable for production use.

For more information, see the [AddressSanitizer
documentation](https://clang.llvm.org/docs/AddressSanitizer.html).

## LeakSanitizer

LeakSanitizer (LSan) detects memory leaks either as part of AddressSanitizer or
as a standalone tool by instrumenting code at runtime to track all memory and
thread management functions (i.e., interceptors), and searching for memory that
remain allocated but are no longer reachable by any references in the program,
at program termination or during specific checkpoints.

LeakSanitizer can detect:

* Memory allocated dynamically (e.g., via `malloc`, `new`) that are not freed or
  deleted and is no longer referenced in the program (i.e., directly leaked
  memory).
* Memory allocated dynamically that is referenced by another memory that are not
  freed or deleted and is no longer referenced in the program (i.e., indirectly
  leaked memory).

LeakSanitizer does not use instrumentation at compile time and works without
recompiling all code using LeakSanitizer.

LeakSanitizer impacts performance due to the interceptors and the checks for
memory leaks at program termination. LeakSanitizer and its runtime are not
suitable for production use.

For more information, see the [LeakSanitizer
documentation](https://clang.llvm.org/docs/LeakSanitizer.html).

## Disclaimer

The quality of the Sanitizers implementation and support varies across operating
systems and architectures, and relies heavily on LLVM implementation--they are
mostly implemented in and supported by LLVM.

Using a different LLVM or runtime version than the one used by the Rust compiler
is not supported. Using Sanitizers in mixed-language binaries (also known as
“mixed binaries”) is supported when the same LLVM and runtime version is used by
all languages.
