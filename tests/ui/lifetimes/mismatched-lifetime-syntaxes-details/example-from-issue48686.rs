#![deny(mismatched_lifetime_syntaxes)]

struct Foo;

impl Foo {
    pub fn get_mut(&'static self, x: &mut u8) -> &mut u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        unsafe { &mut *(x as *mut _) }
    }
}

fn main() {}
