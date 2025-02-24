//@ compile-flags: -g --crate-type=rlib -Cdwarf-version=4

pub fn check_is_even(number: &u64) -> bool {
    number % 2 == 0
}
