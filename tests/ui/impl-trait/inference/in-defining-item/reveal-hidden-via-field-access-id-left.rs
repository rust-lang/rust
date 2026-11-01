// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// known-bug: unknown

struct I {
    field: (),
}
const I: I = I { field: () };

fn id2<T>(_: T, x: T) -> T {
    x
}

fn test(n: bool) -> impl Sized {
    let true = n else { return I };
    let _: () = id2(test(!n), I).field;
    I
}

fn main() {}
