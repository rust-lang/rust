//@ run-rustfix
// Regression test for #135580: check that we do not suggest to simply drop
// the `*` to make the types match when the source is a raw pointer while
// the target type is a reference.

struct S;

fn main() {
    let mut s = S;
    let x = &raw const s;
    let _: &S = unsafe { *x };
    //~^ ERROR mismatched types
    //~| HELP consider borrowing here

    let x = &raw mut s;
    let _: &mut S = unsafe { *x };
    //~^ ERROR mismatched types
    //~| HELP consider mutably borrowing here
}
