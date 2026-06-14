// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

#![feature(type_alias_impl_trait)]

struct W<T>(T);
struct WShow;
impl W<()> {
    pub fn show(&self) -> WShow {
        WShow
    }
}

struct OnWShow;
trait OnW {
    fn show(&self) -> OnWShow {
        OnWShow
    }
}
impl<T> OnW for W<T> {}

fn id2<T>(_: T, x: T) -> T {
    x
}

type Test = impl Sized;

fn define() -> Test {}

fn test(x: Test) {
    let _: WShow = W(id2((), x)).show();
}

fn main() {}
