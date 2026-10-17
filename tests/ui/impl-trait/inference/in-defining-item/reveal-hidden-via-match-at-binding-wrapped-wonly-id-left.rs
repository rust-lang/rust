// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// [new]check-pass
// [old]known-bug: unknown

struct W<T>(T);
struct WShow;
impl W<()> {
    #[allow(dead_code)]
    pub fn show(&self) -> WShow {
        WShow
    }
}

fn id2<T>(_: T, x: T) -> T {
    x
}

fn test(n: bool) -> impl Sized {
    let true = n else { return };
    let _: WShow = match W(id2(test(!n), ())) {
        x @ W(()) => x.show(),
    };
}

fn main() {}
