// edition:2021
// revisions: new old
// [new]compile-flags: -Ztrait-solver=next
// [old]compile-flags: -Ztrait-solver=classic
// known-bug: unknown

struct I {
    field: (),
}
const I: I = I { field: () };

fn test(n: bool) -> impl Sized {
    let true = n else { return I };
    let _: () = test(!n).field;
    I
}

fn main() {}
