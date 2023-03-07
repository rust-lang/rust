// Check that an unsafe impl does not imply that unsafe actions are
// legal in the methods.

// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

unsafe trait UnsafeTrait : Sized {
    fn foo(self) { }
}

unsafe impl UnsafeTrait for *mut isize {
    fn foo(self) {
        // Unsafe actions are not made legal by taking place in an unsafe trait:
        *self += 1;
        //~^ ERROR E0133
    }
}

fn main() { }
