// compile-flags: --test

#![feature(custom_test_frameworks)]
#![deny(unnameable_test_items)]

fn foo() {
    #[test_case]
    //~^ ERROR cannot test inner items [unnameable_test_items]
    fn test2() {}
}
