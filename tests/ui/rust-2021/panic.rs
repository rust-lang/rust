//@ edition:2021

fn main() {
    debug_assert!(false, 123);
    //~^ ERROR must be a string literal
    assert!(false, 123);
    //~^ ERROR must be a string literal
    panic!(false, 123);
    //~^ ERROR must be a string literal

    std::debug_assert!(false, 123);
    //~^ ERROR must be a string literal
    std::assert!(false, 123);
    //~^ ERROR must be a string literal
    std::panic!(false, 123);
    //~^ ERROR must be a string literal

    core::debug_assert!(false, 123);
    //~^ ERROR must be a string literal
    core::assert!(false, 123);
    //~^ ERROR must be a string literal
    core::panic!(false, 123);
    //~^ ERROR must be a string literal
}
