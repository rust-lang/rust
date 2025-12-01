#![allow(incomplete_features)]
#![feature(fn_delegation)]

trait Trait {
    fn static_method2(x: i32, y: i32) -> i32 {
        x + y
    }
}

struct S;
impl Trait for S {}

pub fn main() {
    'foo: loop {
        reuse <S as Trait>::static_method2 { loop { break 'foo; } }
        //~^ ERROR use of unreachable label `'foo`
    }
}
