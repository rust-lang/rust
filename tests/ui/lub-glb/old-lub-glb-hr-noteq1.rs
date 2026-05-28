// Test taking the LUB of two function types that are not equatable but where one is more
// general than the other. Test the case where the more general type (`x`) is the first
// match arm specifically.

//@ revisions: leak noleak
//@[noleak] compile-flags:-Zno-leak-check

fn foo(x: for<'a, 'b> fn(&'a u8, &'b u8) -> &'a u8, y: for<'a> fn(&'a u8, &'a u8) -> &'a u8) {
    // The two types above are not equivalent. With the older LUB/GLB
    // algorithm, this may have worked (I don't remember), but now it
    // doesn't because we require equality.
    let z = match 22 {
        0 => x,
        _ => y,
        //[leak]~^ ERROR `match` arms have incompatible types
        //[noleak]~^^ ERROR mismatched types
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
