#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn static_method(x: i32) -> i32 { x }
}

struct F;

struct S(F);
impl Trait for S {}

fn foo(x: i32) -> i32 { x }

fn bar<T: Default>(_: T) {
    reuse Trait::static_method {
    //~^ ERROR mismatched types
        let _ = T::Default();
        //~^ ERROR can't use generic parameters from outer item
    }
}

fn main() {
    let y = 0;
    reuse <S as Trait>::static_method {
        let x = y;
        //~^ ERROR can't capture dynamic environment in a fn item
        foo(self);

        let reuse_ptr: fn(i32) -> i32  = static_method;
        reuse_ptr(0)
    }
    self.0;
    //~^ ERROR expected value, found module `self`
    let z = x;
    //~^ ERROR cannot find value `x` in this scope
}
