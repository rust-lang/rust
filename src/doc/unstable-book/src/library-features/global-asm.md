# `global_asm`

The tracking issue for this feature is: [#35119]

[#35119]: https://github.com/rust-lang/rust/issues/35119

------------------------

The `global_asm!` macro allows the programmer to write arbitrary
assembly outside the scope of a function body, passing it through
`rustc` and `llvm` to the assembler. That is to say, `global_asm!` is
equivalent to assembling the asm with an external assembler and then
linking the resulting object file with the current crate.

`global_asm!` fills a role not currently satisfied by either `asm!`
or `#[naked]` functions. The programmer has _all_ features of the
assembler at their disposal. The linker will expect to resolve any
symbols defined in the inline assembly, modulo any symbols marked as
external. It also means syntax for directives and assembly follow the
conventions of the assembler in your toolchain.

A simple usage looks like this:

```rust,ignore (requires-external-file)
#![feature(global_asm)]
# // you also need relevant target_arch cfgs
global_asm!(include_str!("something_neato.s"));
```

And a more complicated usage looks like this:

```rust,no_run
#![feature(global_asm)]
# #[cfg(any(target_arch="x86", target_arch="x86_64"))]
# mod x86 {

pub mod sally {
    global_asm!(
        ".global foo",
        "foo:",
        "jmp baz",
    );

    #[no_mangle]
    pub unsafe extern "C" fn baz() {}
}

// the symbols `foo` and `bar` are global, no matter where
// `global_asm!` was used.
extern "C" {
    fn foo();
    fn bar();
}

pub mod harry {
    global_asm!(
        ".global bar",
        "bar:",
        "jmp quux",
    );

    #[no_mangle]
    pub unsafe extern "C" fn quux() {}
}
# }
```

You may use `global_asm!` multiple times, anywhere in your crate, in
whatever way suits you. However, you should not rely on assembler state
(e.g. assembler macros) defined in one `global_asm!` to be available in
another one. It is implementation-defined whether the multiple usages
are concatenated into one or assembled separately.

`global_asm!` also supports `const` operands like `asm!`, which allows
constants defined in Rust to be used in assembly code:

```rust,no_run
#![feature(global_asm, asm_const)]
# #[cfg(any(target_arch="x86", target_arch="x86_64"))]
# mod x86 {
const C: i32 = 1234;
global_asm!(
    ".global bar",
    "bar: .word {c}",
    c = const C,
);
# }
```

The syntax for passing operands is the same as `asm!` except that only
`const` operands are allowed. Refer to the [asm](asm.md) documentation
for more details.

On x86, the assembly code will use intel syntax by default. You can
override this by adding `options(att_syntax)` at the end of the macro
arguments list:

```rust,no_run
#![feature(global_asm, asm_const)]
# #[cfg(any(target_arch="x86", target_arch="x86_64"))]
# mod x86 {
global_asm!("movl ${}, %ecx", const 5, options(att_syntax));
// is equivalent to
global_asm!("mov ecx, {}", const 5);
# }
```

------------------------

If you don't need quite as much power and flexibility as
`global_asm!` provides, and you don't mind restricting your inline
assembly to `fn` bodies only, you might try the
[asm](asm.md) feature instead.
