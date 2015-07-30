- Feature Name: compile_time_asserts
- Start Date: 2015-07-30
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

If the compiler can detect at compile-time that something will always
cause a `debug_assert` or an `assert` it should instead
insert an unconditional runtime-panic and issue a warning.

# Motivation

Expressions are const-evaluated even when they are not in a const environment.

For example

```rust
fn blub<T>(t: T) -> T { t }
let x = 5 << blub(42);
```

will not cause a compiler error currently, while `5 << 42` will.
If the constant evaluator gets smart enough, it will be able to const evaluate
the `blub` function. This would be a breaking change, since the code would not
compile anymore. (this occurred in https://github.com/rust-lang/rust/pull/26848).

GNAT (an Ada compiler) does this already:

```ada
procedure Hello is
  Var: Integer range 15 .. 20 := 21;
begin
  null;
end Hello;
```

The anonymous subtype `Integer range 15 .. 20` only accepts values in `[15, 20]`.
This knowledge is used by GNAT to emit the following warning during compilation:

```
warning: value not in range of subtype of "Standard.Integer" defined at line 2
warning: "Constraint_Error" will be raised at run time
```

I don't have a GNAT with `-emit-llvm` handy, but here's the asm with `-O0`:

```asm
.cfi_startproc
pushq   %rbp
.cfi_def_cfa_offset 16
.cfi_offset 6, -16
movq    %rsp, %rbp
.cfi_def_cfa_register 6
movl    $2, %esi
movl    $.LC0, %edi
movl    $0, %eax
call    __gnat_rcheck_CE_Range_Check
```


# Detailed design

The PRs https://github.com/rust-lang/rust/pull/26848 and https://github.com/rust-lang/rust/pull/25570 will be setting a precedent
for warning about such situations (WIP, not pushed yet).
All future additions to the const-evaluator need to notify the const evaluator
that when it encounters a statically known erroneous situation, the
entire expression must be replaced by a panic and a warning must be emitted.

# Drawbacks

None, if we don't do anything, the const evaluator cannot get much smarter.

# Alternatives

## allow breaking changes

Let the compiler error on things that will unconditionally panic at runtime.

## only warn, don't influence code generation

This has the disadvantage, that in release-mode statically known issues like
overflow or shifting more than the number of bits available will not be
caught even at runtime.

# Unresolved questions

How to implement this?
