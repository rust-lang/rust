// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// known-bug: unknown

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

fn test(n: bool) -> impl OnI {
    let true = n else { return I };
    // This should be a defining use, but is not currently.
    let _: IShow = match test(!n) {
        x @ I => x.show(),
    };
    I
}

fn main() {}
