// Test that the NLL `relate_tys` code correctly deduces that a
// function returning either argument CANNOT be upcast to one
// that returns always its first argument.
//
//@ compile-flags:-Zno-leak-check

fn make_it() -> for<'a> fn(&'a u32, &'a u32) -> &'a u32 {
    panic!()
}

fn foo() {
    let a: for<'a, 'b> fn(&'a u32, &'b u32) -> &'a u32 = make_it();
    //~^ ERROR mismatched types [E0308]
    drop(a);
}

fn bar() {
    // The code path for patterns is mildly different, so go ahead and
    // test that too:
    let _: for<'a, 'b> fn(&'a u32, &'b u32) -> &'a u32 = make_it();
    //~^ ERROR mismatched types [E0308]
}

fn main() {}
