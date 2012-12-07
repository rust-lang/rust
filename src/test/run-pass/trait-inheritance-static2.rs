trait MyEq { }

trait MyNum {
    static fn from_int(int) -> self;
}

pub trait NumExt: MyEq MyNum { }

struct S { v: int }

impl S: MyEq { }

impl S: MyNum {
    static fn from_int(i: int) -> S {
        S {
            v: i
        }
    }
}

impl S: NumExt { }

fn greater_than_one<T:NumExt>() -> T { from_int(1) }

fn main() {
    let v: S = greater_than_one();
    assert v.v == 1;
}
