#![feature(fn_delegation)]

mod original_ice {
    trait Trait {
        fn foo(&self) -> u8 { 0 }
        fn bar(&self) -> u8 { 0 }
        fn baz(&self) -> u8 { 0 }
    }

    impl Trait for u8 {}
    impl Trait for Rc<Box<dyn Trait>> {}

    struct S(u8);

    use std::rc::Rc;
    impl Trait for S {
        reuse <dyn Trait as Trait>::foo { &self.0 }
        reuse <dyn Trait as Trait>::bar { self.0 }
        reuse <Rc<Box<dyn Trait>> as Trait>::baz { self.0 }
        //~^ ERROR: mismatched types [E0308]
    }
}

mod test_2 {
    trait Trait {
        fn foo(&self) -> u8 { 0 }
    }

    impl Trait for () {}
    impl Trait for u8 {}

    struct S(u8);

    impl Trait for S {
        reuse <() as Trait>::foo { &self.0 }
        //~^ ERROR: mismatched types [E0308]
    }
}

fn main() {}
