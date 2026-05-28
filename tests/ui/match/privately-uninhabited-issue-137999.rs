//@ edition:2024
//@ check-fail

mod m {
    enum Void {}

    pub struct Internal {
        _v: Void,
    }

    pub enum Test {
        A(u32, u32),
        B(Internal),
    }
}

use m::Test;

pub fn f1(x: &mut Test) {
    let r1: &mut u32 = match x {
        Test::A(a, _) => a,
        _ => todo!(),
    };

    let r2: &mut u32 = match x { //~ ERROR cannot use `*x` because it was mutably borrowed
        Test::A(_, b) => b,
        _ => todo!(),
    };

    let _ = *r1;
    let _ = *r2;
}

pub fn f2(x: &mut Test) {
    let r = &mut *x;
    match x { //~ ERROR cannot use `*x` because it was mutably borrowed
        Test::A(_, _) => {}
        _ => {}
    }

    let _ = r;
}

fn main() {}
