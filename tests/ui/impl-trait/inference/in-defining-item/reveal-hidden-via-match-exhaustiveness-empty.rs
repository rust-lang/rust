// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// known-bug: #116821

enum E {}

fn id2<T>(_: T, x: T) -> T {
    x
}

fn test(n: bool, e: E) -> impl Sized {
    let true = n else { return e };
    match test(!n, e) {}
    e
}

fn main() {}
