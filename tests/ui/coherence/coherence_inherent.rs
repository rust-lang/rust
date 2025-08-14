// Tests that methods that implement a trait cannot be invoked
// unless the trait is imported.

mod Lib {
    pub trait TheTrait {
        fn the_fn(&self);
    }

    pub struct TheStruct;

    impl TheTrait for TheStruct {
        fn the_fn(&self) {}
    }
}

mod Import {
    // Trait is in scope here:
    use crate::Lib::TheStruct;
    use crate::Lib::TheTrait;

    fn call_the_fn(s: &TheStruct) {
        s.the_fn();
    }
}

mod NoImport {
    // Trait is not in scope here:
    use crate::Lib::TheStruct;

    fn call_the_fn(s: &TheStruct) {
        s.the_fn();
        //~^ ERROR E0599
    }
}

fn main() {}
