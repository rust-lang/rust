// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// check-pass

// Make sure that the compiler can handle `ReErased` in the hidden type of an opaque.

fn foo<'a: 'a>(x: &'a Vec<i32>) -> impl Fn() + 'static {
    || ()
}

fn bar() -> impl Fn() + 'static {
    foo(&vec![])
}

fn main() {}
