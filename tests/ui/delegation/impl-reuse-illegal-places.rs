#![allow(incomplete_features)]
#![feature(fn_delegation)]

trait T {
    fn f(&self) {}
}

struct S;
impl T for S {}

struct F(S);

struct X {
    reuse impl T for F { self.0 }
    //~^ ERROR expected `:`, found keyword `impl`
}

impl X {
    reuse impl T for F { self.0 }
    //~^ ERROR implementation is not supported in `trait`s or `impl`s
}

trait Trait {
    reuse impl T for F { self.0 }
    //~^ ERROR implementation is not supported in `trait`s or `impl`s
}

extern "C" {
    reuse impl T for F { self.0 }
    //~^ ERROR implementation is not supported in `extern` blocks
}

mod m {
    mod inner {
        pub fn foo() {}
    }

    reuse inner::{
        reuse impl T for F { self.0 }
        //~^ ERROR expected identifier, found keyword `impl`
        //~| ERROR expected one of `,`, `as`, or `}`, found keyword `impl`
        //~| ERROR expected one of `,`, `as`, or `}`, found `T`
        //~| ERROR expected identifier, found keyword `for`
        //~| ERROR expected one of `,`, `as`, or `}`, found keyword `for`
        //~| ERROR expected one of `,`, `as`, or `}`, found `F`
        //~| ERROR expected one of `,`, `as`, or `}`, found `{`
    }
}
//~^ ERROR expected item, found `}`

fn main() {}
