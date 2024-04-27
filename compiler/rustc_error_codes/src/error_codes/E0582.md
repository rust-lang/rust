A lifetime is only present in an associated-type binding, and not in the input
types to the trait.

Erroneous code example:

```compile_fail,E0582
fn bar<F>(t: F)
    // No type can satisfy this requirement, since `'a` does not
    // appear in any of the input types (here, `i32`):
    where F: for<'a> Fn(i32) -> Option<&'a i32>
{
}

fn main() { }
```

To fix this issue, either use the lifetime in the inputs, or use
`'static`. Example:

```
fn bar<F, G>(t: F, u: G)
    where F: for<'a> Fn(&'a i32) -> Option<&'a i32>,
          G: Fn(i32) -> Option<&'static i32>,
{
}

fn main() { }
```

Note: The examples above used to be (erroneously) accepted by the
compiler, but this was since corrected. See [issue #33685] for more
details.

[issue #33685]: https://github.com/rust-lang/rust/issues/33685
