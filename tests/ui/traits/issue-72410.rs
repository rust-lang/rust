// Regression test for #72410, this should be used with debug assertion enabled.

// should be fine
pub trait Foo {
    fn map()
    where
        Self: Sized,
        for<'a> &'a mut [u8]: ;
}

// should fail
pub trait Bar {
    fn map()
    where for<'a> &'a mut [dyn Bar]: ;
    //~^ ERROR: the trait `Bar` cannot be made into an object
}

fn main() {}
