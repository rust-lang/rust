// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

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

fn test(n: bool) -> impl Sized {
    let true = n else { return };
    let _: WShow = W(id2((), test(!n))).show();
}

fn main() {}
