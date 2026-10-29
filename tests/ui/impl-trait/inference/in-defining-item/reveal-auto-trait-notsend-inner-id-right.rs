// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

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
    let true = n else { return &() as *const () };
    let _: OnWShow = (&&W(id2(&() as *const (), test(!n)))).show();
    &() as *const ()
}

fn main() {}
