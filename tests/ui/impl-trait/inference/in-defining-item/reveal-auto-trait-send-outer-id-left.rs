// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// [new]check-pass
// [old]known-bug: unknown

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

fn test(n: bool) -> impl Sized {
    let true = n else { return };
    let _: OnWSendShow = id2(&&W(test(!n)), &&W(())).show();
}

fn main() {}
