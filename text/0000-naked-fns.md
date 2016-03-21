- Feature Name: `naked_fns`
- Start Date: 2015-07-10
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add support for generating naked (prologue/epilogue-free) functions via a new
function attribute.

# Motivation

Some systems programming tasks require that the programmer have complete control
over function stack layout and interpretation, generally in cases where the
compiler lacks support for a specific use case. While these cases can be
addressed by building the requisite code with external tools and linking with
Rust, it is advantageous to allow the Rust compiler to drive the entire process,
particularly in that code may be generated via monomorphization or macro
expansion.

When writing interrupt handlers for example, most systems require additional
state be saved beyond the usual ABI requirements.  To avoid corrupting program
state, the interrupt handler must save the registers which might be modified
before handing control to compiler-generated code. Consider a contrived
interrupt handler for x86\_64:

```rust
unsafe fn isr_nop() {
    asm!("push %rax"
         /* Additional pushes elided */ :::: "volatile");
    let n = 0u64;
    asm!("pop %rax"
         /* Additional pops elided */ :::: "volatile");
}
```

The generated assembly for this function might resemble the following
(simplified for readability):

```x86
isr_nop:
    sub $8, %rsp
    push %rax
    movq $0, 0(%rsp)
    pop %rax
    add $8, %rsp
    retq
```

Here the programmer's need to save machine state conflicts with the compiler's
assumption that it has complete control over stack layout, with the result that
the saved value of `rax` is clobbered by the compiler. Given that details of
stack layout for any given function are not predictable (and may change with
compiler version or optimization settings), attempting to predict the stack
layout to sidestep this issue is infeasible.

When interacting with FFIs that are not natively supported by the compiler,
a similar situation arises where the programmer knows the expected calling
convention and can implement a translation between the foreign ABI and one
supported by the compiler.

Support for naked functions also allows programmers to write functions that
would otherwise be unsafe, such as the following snippet which returns the
address of its caller when called with the C ABI on x86.

```
    mov 4(%ebp), %eax
    ret
```

---

Because the compiler depends on a function prologue and epilogue to maintain
storage for local variable bindings, it is generally unsafe to write anything
but inline assembly inside a naked function.  The [LLVM language
reference](http://llvm.org/docs/LangRef.html#function-attributes) describes this
feature as having "very system-specific consequences", which the programmer must
be aware of.

# Detailed design

Add a new function attribute to the language, `#[naked]`, indicating the
function should have prologue/epilogue emission disabled.

Because the calling convention of a naked function is not guaranteed to match
any calling convention the compiler is compatible with, calls to naked functions
from within Rust code are forbidden unless the function is also declared with
a well-defined ABI.

Defining a naked function with the default (Rust) ABI is an error, because the
Rust ABI is unspecified and the programmer can never write a function which is
guaranteed to be compatible. For example, The function declaration of `foo` in
the following code block is an error.

```rust
#[naked]
unsafe fn foo() { }
```

The following variant is not an error because the C calling convention is
well-defined and it is thus possible for the programmer to write a conforming
function:

```rust
#[naked]
extern "C" fn foo() { }
```

---

Because the compiler cannot verify the correctness of code written in a naked
function (since it may have an unknown calling convention), naked functions must
be declared `unsafe` or contain no non-`unsafe` statements in the body. The
function `error` in the following code block is a compile-time error, whereas
the functions `correct1` and `correct2` are permitted.

```
#[naked]
extern "C" fn error(x: &mut u8) {
    *x += 1;
}

#[naked]
unsafe extern "C" fn correct1(x: &mut u8) {
    *x += 1;
}

#[naked]
extern "C" fn correct2() {
    unsafe {
        *x += 1;
    }
}
```

## Example

The following example illustrates the possible use of a naked function for
implementation of an interrupt service routine on 32-bit x86.

```rust
use std::intrinsics;
use std::sync::atomic::{self, AtomicUsize, Ordering};

#[naked]
#[cfg(target_arch="x86")]
unsafe fn isr_3() {
    asm!("pushad
          call increment_breakpoint_count
          popad
          iretd" :::: "volatile");
    intrinsics::unreachable();
}

static bp_count: AtomicUsize = ATOMIC_USIZE_INIT;

#[no_mangle]
pub fn increment_breakpoint_count() {
    bp_count.fetch_add(1, Ordering::Relaxed);
}

fn register_isr(vector: u8, handler: fn() -> ()) { /* ... */ }

fn main() {
    register_isr(3, isr_3);
    // ...
}
```

## Implementation Considerations

The current support for `extern` functions in `rustc` generates a minimum of two
basic blocks for any function declared in Rust code with a non-default calling
convention: a trampoline which translates the declared calling convention to the
Rust convention, and a Rust ABI version of the function containing the actual
implementation. Calls to the function from Rust code call the Rust ABI version
directly.

For naked functions, it is impossible for the compiler to generate a Rust ABI
version of the function because the implementation may depend on the calling
convention. In cases where calling a naked function from Rust is permitted, the
compiler must be able to use the target calling convention directly rather than
call the same function with the Rust convention.

# Drawbacks

The utility of this feature is extremely limited to most users, and it might be
misused if the implications of writing a naked function are not carefully
considered.

# Alternatives

Do nothing. The required functionality for the use case outlined can be
implemented outside Rust code and linked in as needed. Support for additional
calling conventions could be added to the compiler as needed, or emulated with
external libraries such as `libffi`.

# Unresolved questions

It is easy to quietly generate wrong code in naked functions, such as by causing
the compiler to allocate stack space for temporaries where none were
anticipated. There is currently no restriction on writing Rust statements inside
a naked function, while most compilers supporting similar features either
require or strongly recommend that authors write only inline assembly inside
naked functions to ensure no code is generated that assumes a particular stack
layout. It may be desirable to place further restrictions on what statements are
permitted in the body of a naked function, such as permitting only `asm!`
statements.

The `unsafe` requirement on naked functions may not be desirable in all cases.
However, relaxing that requirement in the future would not be a breaking change.

Because a naked function may use a calling convention unknown to the compiler,
it may be useful to add a "unknown" calling convention to the compiler which is
illegal to call directly. Absent this feature, functions implementing an unknown
ABI would need to be declared with a calling convention which is known to be
incorrect and depend on the programmer to avoid calling such a function
incorrectly since it cannot be prevented statically.
