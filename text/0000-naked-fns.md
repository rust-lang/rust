- Feature Name: `naked_fns`
- Start Date: 2015-07-10
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add support for generating naked (prologue/epilogue-free) functions via a new
function attribute.

# Motivation

Some systems programming tasks require that machine state not be modified at all
on function entry so it can be preserved- particularly in interrupt handlers.
For example, x86\_64 preserves only the stack pointer, flags register, and
instruction pointer on interrupt entry. To avoid corrupting program state, the
interrupt handler must save the registers which might be modified before handing
control to compiler-generated code. Consider a contrived interrupt handler:

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

In other languages (particularly C), "naked" functions omit the prologue and
epilogue (represented by the modifications to `rsp` in the above example) to
allow the programmer complete control over stack layout. This makes the
availability of stack space for compiler use unpredictable, usually implying
that the body of such a function must consist entirely of inline assembly
statements (such as a jump or call to another function).

The [LLVM language
reference](http://llvm.org/docs/LangRef.html#function-attributes) describes this
feature as having "very system-specific consequences", which the programmer must
be aware of.

# Detailed design

Add a new function attribute to the language, `#[naked]`, indicating the
function should have prologue/epilogue emission disabled.

For example, the following construct could be assumed not to generate extra code
on entry to `isr_caller` which might violate the programmer's assumptions, while
allowing the compiler to generate the function definition as usual:

```rust
#[naked]
unsafe fn isr_caller() {
    asm!("push %rax
          call other_function
          pop %rax
          iretq" :::: "volatile");
    core::intrinsics::unreachable();
}

#[no_mangle]
pub fn other_function() {

}
```

# Drawbacks

The utility of this feature is extremely limited to most users, and it might be
misused if the implications of writing a naked function are not carefully
considered.

# Alternatives

Do nothing. The required functionality for the use case outlined can be
implemented outside Rust code (such as with a small amount of externally-built
assembly) and merely linked in as needed.

Add a new calling convention (`extern "interrupt" fn ...`) which is defined to
do any necessary state saving for interrupt service routines. This permits more
efficient code to be generated for the motivating example (omitting a 'call'
instruction which is necessary for any non-trivial ISR), but may not be
appropriate for other situations that might call for a naked function.
Implementation of additional calling conventions like this in the current
`rustc` would involve significant modification to LLVM to support it (whereas
the proof-of-concept patch for `#[naked]` is less than 10 lines of code).

# Unresolved questions

It is easy to quietly generate wrong code in naked functions, such as by causing
the compiler to allocate stack space for temporaries where none were
anticipated. It may be desirable to allow the `#[naked]` attribute on `unsafe`
functions only, reinforcing the need for extreme care in the use of this
feature.
