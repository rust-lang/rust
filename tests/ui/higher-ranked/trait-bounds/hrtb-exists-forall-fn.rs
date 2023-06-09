// Test an `exists<'a> { forall<'b> { 'a = 'b } }` pattern -- which should not compile!
//
// In particular, we test this pattern in trait solving, where it is not connected
// to any part of the source code.

fn foo<'a>() -> fn(&'a u32) {
    panic!()
}

fn main() {
    // Here, proving that `fn(&'a u32) <: for<'b> fn(&'b u32)`:
    //
    // - instantiates `'b` with a placeholder `!b`,
    // - requires that `&!b u32 <: &'a u32` and hence that `!b: 'a`,
    // - but we can never know this.

    let _: for<'b> fn(&'b u32) = foo(); //~ ERROR mismatched types
}
