# `export-executable-symbols`

The tracking issue for this feature is: [#84161](https://github.com/rust-lang/rust/issues/84161).

------------------------

The `-Zexport-executable-symbols` compiler flag makes `rustc` export symbols from executables. The resulting binary is runnable, but can also be used as a dynamic library. This is useful for interoperating with programs written in other languages, in particular languages with a runtime like Java or Lua.

For example on windows:
```rust
#[no_mangle]
fn my_function() -> usize {
    return 42;
}

fn main() {
    println!("Hello, world!");
}
```

A standard `cargo build` will produce a `.exe` without an export directory. When the `export-executable-symbols` flag is added

```Bash
export RUSTFLAGS="-Zexport-executable-symbols"
cargo build
```

the binary has an export directory with the functions:

```plain
The Export Tables (interpreted .edata section contents)

...

[Ordinal/Name Pointer] Table
    [   0] my_function
    [   1] main
```
(the output of `objdump -x` on the binary)

Please note that the `#[no_mangle]` attribute is required. Without it, the symbol is not exported.

The equivalent of this flag in C and C++ compilers is the `__declspec(dllexport)` annotation or the `-rdynamic` linker flag.
