Abstract return types (written `impl Trait` for some trait `Trait`) are only
allowed as function and inherent impl return types.

Erroneous code example:

```compile_fail,E0562
fn main() {
    let count_to_ten: impl Iterator<Item=usize> = 0..10;
    // error: `impl Trait` not allowed outside of function and inherent method
    //        return types
    for i in count_to_ten {
        println!("{}", i);
    }
}
```

Make sure `impl Trait` only appears in return-type position.

```
fn count_to_n(n: usize) -> impl Iterator<Item=usize> {
    0..n
}

fn main() {
    for i in count_to_n(10) {  // ok!
        println!("{}", i);
    }
}
```

See [RFC 1522] for more details.

[RFC 1522]: https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md
