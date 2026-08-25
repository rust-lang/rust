//@ run-pass
pub trait MyEq {}

pub trait MyNum {
    fn from_int(_: isize) -> Self;
}

pub trait NumExt: MyEq + MyNum { }

struct S { v: isize }

impl MyEq for S { }

impl MyNum for S {
    fn from_int(i: isize) -> S {
        S {
            v: i
        }
    }
}

impl NumExt for S { }

fn greater_than_one<T:NumExt>() -> T { MyNum::from_int(1) }

pub fn main() {
    let v: S = greater_than_one();
    assert_eq!(v.v, 1);
}
