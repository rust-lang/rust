#![deny(elided_named_lifetimes)]

struct Foo;

impl Foo {
    pub fn get_mut(&'static self, x: &mut u8) -> &mut u8 {
        //~^ ERROR elided lifetime has a name
        unsafe { &mut *(x as *mut _) }
    }
}

fn main() {}
