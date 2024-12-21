//! This test case is modified from <https://github.com/rust-lang/rust/issues/93775>.
//! The type printing involves recursive calls that lead to stack overflow.
//! If it no longer crashes, please increase the nested type levels
//! unless you are fixing this issue.
//@ known-bug: #93775

#![recursion_limit = "2049"]

use std::marker::PhantomData;

struct Z;
struct S<T>(PhantomData<T>);

type Nested4<T> = S<S<S<S<T>>>>;
type Nested16<T> = Nested4<Nested4<Nested4<Nested4<T>>>>;
type Nested64<T> = Nested16<Nested16<Nested16<Nested16<T>>>>;
type Nested256<T> = Nested64<Nested64<Nested64<Nested64<T>>>>;
type Nested1024<T> = Nested256<Nested256<Nested256<Nested256<T>>>>;
type Nested2048<T> = Nested1024<Nested1024<T>>;

type Nested = Nested2048<Z>;

trait AsNum {
    const NUM: u32;
}

impl AsNum for Z {
    const NUM: u32 = 0;
}

impl<T: AsNum> AsNum for S<T> {
    const NUM: u32 = T::NUM + 1;
}

fn main() {
    let _ = Nested::NUM;
}
