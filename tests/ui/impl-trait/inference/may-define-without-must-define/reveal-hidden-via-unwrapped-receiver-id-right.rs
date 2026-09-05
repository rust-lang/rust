// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

#![feature(type_alias_impl_trait)]

struct I;
struct IShow;
impl I {
    pub fn show(&self) -> IShow {
        IShow
    }
}

struct OnIShow;
trait OnI {
    fn show(&self) -> OnIShow {
        OnIShow
    }
}
impl OnI for I {}

fn id2<T>(_: T, x: T) -> T {
    x
}

type Test = impl Sized;

fn define() -> Test {
    I
}

fn test(x: Test) {
    let _: IShow = id2(I, x).show();
}

fn main() {}
