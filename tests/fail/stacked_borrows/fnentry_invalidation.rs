// Test that spans displayed in diagnostics identify the function call, not the function
// definition, as the location of invalidation due to FnEntry retag. Technically the FnEntry retag
// occurs inside the function, but what the user wants to know is which call produced the
// invalidation.
fn main() {
    let mut x = 0i32;
    let z = &mut x as *mut i32;
    x.do_bad();
    unsafe {
        let _oof = *z; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}

trait Bad {
    fn do_bad(&mut self) {
        // who knows
    }
}

impl Bad for i32 {}
