fn foo(_s: i16) { }

fn bar(_s: u32) { }

fn main() {
    foo(1*(1 as isize));
    //~^ ERROR mismatched types
    //~| expected i16, found isize

    bar(1*(1 as usize));
    //~^ ERROR mismatched types
    //~| expected u32, found usize
}
