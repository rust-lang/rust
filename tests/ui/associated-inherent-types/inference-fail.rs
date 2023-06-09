#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl<T> S<T> { type P = (); }

fn main() {
    // There is no way to infer this type.
    let _: S<_>::P = (); //~ ERROR type annotations needed
}
