//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] run-rustfix
fn main() {
    let _ = (-10..=10).find(|x: i32| x.signum() == 0);
    //[current]~^ ERROR type mismatch in closure arguments
    //[next]~^^ ERROR expected a `FnMut(&<std::ops::RangeInclusive<{integer}> as Iterator>::Item)` closure, found
    let _ = (-10..=10).find(|x: &&&i32| x.signum() == 0);
    //[current]~^ ERROR type mismatch in closure arguments
    //[next]~^^ ERROR expected `RangeInclusive<{integer}>` to be an iterator that yields `&&i32`, but it yields `{integer}`
    //[next]~| ERROR expected a `FnMut(&<std::ops::RangeInclusive<{integer}> as Iterator>::Item)` closure, found
}
