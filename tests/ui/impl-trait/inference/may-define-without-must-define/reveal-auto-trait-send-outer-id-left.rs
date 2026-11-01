// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// known-bug: unknown

#![feature(type_alias_impl_trait)]

struct W<T>(T);

struct OnWShow;
trait OnW {
    fn show(&self) -> OnWShow {
        OnWShow
    }
}

struct OnWSendShow;
trait OnWSend {
    fn show(&self) -> OnWSendShow {
        OnWSendShow
    }
}

impl<T> OnW for W<T> {}
impl<T: Send> OnWSend for &W<T> {}

fn id2<T>(_: T, x: T) -> T {
    x
}

type Test = impl Sized;

fn define() -> Test {}

fn test(x: Test) {
    let _: OnWSendShow = id2(&&W(x), &&W(())).show();
}

fn main() {}
