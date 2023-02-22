// known-bug: unknown

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct S<T>(T);

impl S<()> {
    type P = i128;
}

fn main() {
    // We fail to infer `_ == ()` here.
    let _: S<_>::P;
}
