// Test taking the LUB of two function types that are not equatable but where
// one is more general than the other. Test the case where the more general type
// (`x`) is the second match arm specifically.
//
// FIXME(#73154) Pure NLL checker without leak check accepts this test.
// (Note that it still errors in old-lub-glb-hr-noteq1.rs). What happens
// is that, due to the ordering of the match arms, we pick the correct "more
// general" fn type, and we ignore the errors from the non-NLL type checker that
// requires equality. The NLL type checker only requires a subtyping
// relationship, and that holds. To unblock landing NLL - and ensure that we can
// choose to make this always in error in the future - we perform the leak check
// after coercing a function pointer.

//@ revisions: leak noleak
//@[noleak] compile-flags: -Zno-leak-check

//@[noleak] check-pass

fn foo(x: for<'a, 'b> fn(&'a u8, &'b u8) -> &'a u8, y: for<'a> fn(&'a u8, &'a u8) -> &'a u8) {
    // The two types above are not equivalent. With the older LUB/GLB
    // algorithm, this may have worked (I don't remember), but now it
    // doesn't because we require equality.
    let z = match 22 {
        0 => y,
        _ => x,
        //[leak]~^ ERROR `match` arms have incompatible types
    };
}

fn foo_cast(x: for<'a, 'b> fn(&'a u8, &'b u8) -> &'a u8, y: for<'a> fn(&'a u8, &'a u8) -> &'a u8) {
    // But we can *upcast* explicitly the type of `x` and figure
    // things out:
    let z = match 22 {
        0 => x as for<'a> fn(&'a u8, &'a u8) -> &'a u8,
        _ => y,
    };
}

fn main() {}
