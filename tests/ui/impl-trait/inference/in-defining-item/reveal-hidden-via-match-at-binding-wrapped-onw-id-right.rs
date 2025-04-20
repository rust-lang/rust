// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

struct I;

struct W<T>(T);
struct WIShow;
impl W<I> {
    pub fn show(&self) -> WIShow {
        WIShow
    }
}

struct OnWShow;
trait OnW {
    fn show(&self) -> OnWShow {
        OnWShow
    }
}
impl<T: Sized> OnW for W<T> {}

fn id2<T>(_: T, x: T) -> T {
    x
}

fn test(n: bool) -> impl Sized {
    let true = n else { return I };
    let _: WIShow = match W(id2(I, test(!n))) {
        x @ W(I) => x.show(),
    };
    I
}

fn main() {}
