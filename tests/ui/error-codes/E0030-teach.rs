//@ compile-flags: -Z teach

fn main() {
    match 5u32 {
        1000 ..= 5 => {}
        //~^ ERROR lower range bound must be less than or equal to upper
    }
}
