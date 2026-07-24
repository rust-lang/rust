// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// check-pass

fn id2<T>(_: T, x: T) -> T {
    x
}

fn test(n: bool) -> impl Sized {
    let true = n else { return };
    match id2(test(!n), ()) {
        () => (),
    };
}

fn main() {}
