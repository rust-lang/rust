#![feature(fn_delegation)]

trait Trait1 {
    reuse trait_foo_reused as foo;
    //~^ ERROR: encountered a cycle during delegation signature resolution
}

impl Trait1 for () {}

struct S1<T>(T);
impl<T: Trait1> S1<T> {
    reuse Trait1::foo { self.0 }
    //~^ ERROR: encountered a cycle during delegation signature resolution
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

struct S2(S1<()>);
impl S2 {
    reuse S1::<()>::foo { self.0 }
    //~^ ERROR: encountered a cycle during delegation signature resolution
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

reuse S2::foo;
//~^ ERROR: encountered a cycle during delegation signature resolution

struct S3;
impl S3 {
    reuse foo;
    //~^ ERROR: encountered a cycle during delegation signature resolution
}

impl Trait1 for S3 {
    reuse S2::foo { S2(S1(())) }
    //~^ ERROR: encountered a cycle during delegation signature resolution
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

trait Trait2 {
    reuse <S3 as Trait1>::foo { S3 }
    //~^ ERROR: encountered a cycle during delegation signature resolution
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

reuse Trait2::foo as trait_foo;
//~^ ERROR: encountered a cycle during delegation signature resolution
//~| ERROR: type annotations needed

struct S4;
impl S4 {
    reuse trait_foo;
    //~^ ERROR: encountered a cycle during delegation signature resolution
}

reuse S4::trait_foo as trait_foo_reused;
//~^ ERROR: encountered a cycle during delegation signature resolution

fn main() {}
