# `global_asm`

The tracking issue for this feature is: [#35119]

[#35119]: https://github.com/rust-lang/rust/issues/35119

------------------------

The `global_asm!` macro allows the programmer to write arbitrary
assembly outside the scope of a function body, passing it through
`rustc` and `llvm` to the assembler. The macro is a no-frills
interface to LLVM's concept of [module-level inline assembly]. That is,
all caveats applicable to LLVM's module-level inline assembly apply
to `global_asm!`.

[module-level inline assembly]: http://llvm.org/docs/LangRef.html#module-level-inline-assembly

`global_asm!` fills a role not currently satisfied by either `asm!`
or `#[naked]` functions. The programmer has _all_ features of the
assembler at their disposal. The linker will expect to resolve any
symbols defined in the inline assembly, modulo any symbols marked as
external. It also means syntax for directives and assembly follow the
conventions of the assembler in your toolchain.

A simple usage looks like this:

```rust,ignore
# #![feature(global_asm)]
# you also need relevant target_arch cfgs
global_asm!(include_str!("something_neato.s"));
```

And a more complicated usage looks like this:

```rust,ignore
# #![feature(global_asm)]
# #![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

pub mod sally {
    global_asm!(r#"
        .global foo
      foo:
        jmp baz
    "#);

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
    global_asm!(r#"
        .global bar
      bar:
        jmp quux
    "#);

    #[no_mangle]
    pub unsafe extern "C" fn quux() {}
}
```

You may use `global_asm!` multiple times, anywhere in your crate, in
whatever way suits you. The effect is as if you concatenated all
usages and placed the larger, single usage in the crate root.

------------------------

If you don't need quite as much power and flexibility as
`global_asm!` provides, and you don't mind restricting your inline
assembly to `fn` bodies only, you might try the
[asm](asm.md) feature instead.
