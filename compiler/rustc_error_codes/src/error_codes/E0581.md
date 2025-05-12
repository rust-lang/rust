In a `fn` type, a lifetime appears only in the return type
and not in the arguments types.

Erroneous code example:

```compile_fail,E0581
fn main() {
    // Here, `'a` appears only in the return type:
    let x: for<'a> fn() -> &'a i32;
}
```

The problem here is that the lifetime isn't constrained by any of the arguments,
making it impossible to determine how long it's supposed to live.

To fix this issue, either use the lifetime in the arguments, or use the
`'static` lifetime. Example:

```
fn main() {
    // Here, `'a` appears only in the return type:
    let x: for<'a> fn(&'a i32) -> &'a i32;
    let y: fn() -> &'static i32;
}
```

Note: The examples above used to be (erroneously) accepted by the
compiler, but this was since corrected. See [issue #33685] for more
details.

[issue #33685]: https://github.com/rust-lang/rust/issues/33685
