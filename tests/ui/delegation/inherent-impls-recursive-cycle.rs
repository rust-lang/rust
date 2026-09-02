#![feature(fn_delegation)]

trait Trait1 {
    reuse trait_foo_reused as foo;
}

impl Trait1 for () {}

struct S1<T>(T);
impl<T: Trait1> S1<T> {
    reuse Trait1::foo { self.0 }
}

struct S2(S1<()>);
impl S2 {
    reuse S1::<()>::foo { self.0 }
}

reuse S2::foo;

struct S3;
impl S3 {
    reuse foo;
}

impl Trait1 for S3 {
    reuse S2::foo { S2(S1(())) }
}

trait Trait2 {
    reuse <S3 as Trait1>::foo { S3 }
}

reuse Trait2::foo as trait_foo;

struct S4;
impl S4 {
//~^ ERROR: cycle detected when collecting associated items of `<impl at $DIR/inherent-impls-recursive-cycle.rs:37:1: 37:8>` [E0391]
    reuse trait_foo;
}

reuse S4::trait_foo as trait_foo_reused;

fn main() {}
