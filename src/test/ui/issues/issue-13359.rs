fn foo(_s: i16) { }

fn bar(_s: u32) { }

fn main() {
    foo(1*(1 as isize));
    //~^ ERROR arguments to this function are incorrect
    //~| expected `i16`, found `isize`

    bar(1*(1 as usize));
    //~^ ERROR arguments to this function are incorrect
    //~| expected `u32`, found `usize`
}
