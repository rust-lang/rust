//@ compile-flags: -g --crate-type=rlib -Zdwarf-version=4

pub fn check_is_even(number: &u64) -> bool {
    number % 2 == 0
}
