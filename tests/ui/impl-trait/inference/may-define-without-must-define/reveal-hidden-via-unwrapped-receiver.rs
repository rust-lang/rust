// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// known-bug: unknown

#![feature(type_alias_impl_trait)]

struct I;
struct IShow;
impl I {
    #[allow(dead_code)]
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

type Test = impl Sized;

fn define() -> Test {
    I
}

fn test(x: Test) {
    let _: IShow = x.show();
}

fn main() {}
