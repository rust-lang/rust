//! Check that we consider `Bar<impl Sized>` to successfully unify
//! with both `Bar<u32>` and `Bar<i32>` (in isolation), so we bail
//! out with ambiguity.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver

struct Bar<T>(T);

impl Bar<u32> {
    fn bar(self) {}
}

impl Bar<i32> {
    fn bar(self) {}
}

fn foo(x: bool) -> Bar<impl Sized> {
    if x {
        let x = foo(false);
        x.bar();
        //~^ ERROR: multiple applicable items in scope
    }
    todo!()
}

fn main() {}
