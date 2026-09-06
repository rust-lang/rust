// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// [new]check-pass
// [old]known-bug: unknown

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

type Test = impl Sized;

fn define() -> Test {}

fn test(x: Test) {
    let _: WShow = W(x).show();
}

fn main() {}
