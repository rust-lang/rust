//@ run-rustfix

fn main() {
    for _i 0..2 {} //~ ERROR missing `in`
    for _i of 0..2 {} //~ ERROR missing `in`
    for _i = 0..2 {} //~ ERROR missing `in`
}
