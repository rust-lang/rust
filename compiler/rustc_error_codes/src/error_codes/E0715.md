An `impl` for a `#[marker]` trait tried to override an associated item.

Erroneous code example:

```compile_fail,E0715
#![feature(marker_trait_attr)]

#[marker]
trait Marker {
    const N: usize = 0;
    fn do_something() {}
}

struct OverrideConst;
impl Marker for OverrideConst { // error!
    const N: usize = 1;
}
# fn main() {}
```

Because marker traits are allowed to have multiple implementations for the same
type, it's not allowed to override anything in those implementations, as it
would be ambiguous which override should actually be used.
