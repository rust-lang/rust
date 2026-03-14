// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

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

fn test(n: bool) -> impl Sized {
    let true = n else { return (I,) };
    let _: IShow = match test(!n) {
        (x,) => x.show(),
    };
    (I,)
}

fn main() {}
