#![feature(fn_delegation)]

trait Trait1 {
    reuse trait_foo_reused as foo;
}

impl Trait1 for () {}

struct S1<T>(T);
impl<T: Trait1> S1<T> {
    reuse Trait1::foo { self.0 }
    //~^ ERROR: delegation's target expression is specified for function with no params
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

struct S2(S1<()>);
impl S2 {
    reuse S1::<()>::foo { self.0 }
    //~^ ERROR: cannot find function `foo` in `S1`
}

reuse S2::foo;
//~^ ERROR: cannot find function `foo` in `S2`

struct S3;
impl S3 {
    reuse foo;
}

impl Trait1 for S3 {
    reuse S2::foo { S2(S1(())) }
    //~^ ERROR: delegation's target expression is specified for function with no params
    //~| ERROR: cannot find function `foo` in `S2`
}

trait Trait2 {
    reuse <S3 as Trait1>::foo { S3 }
    //~^ ERROR: delegation's target expression is specified for function with no params
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

reuse Trait2::foo as trait_foo;

struct S4;
impl S4 {
    reuse trait_foo;
}

reuse S4::trait_foo as trait_foo_reused;
//~^ ERROR: cannot find function `trait_foo` in `S4`

fn main() {}
