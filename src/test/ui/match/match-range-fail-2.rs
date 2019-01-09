#![feature(exclusive_range_pattern)]

fn main() {
    match 5 {
        6 ..= 1 => { }
        _ => { }
    };
    //~^^^ ERROR lower range bound must be less than or equal to upper

    match 5 {
        0 .. 0 => { }
        _ => { }
    };
    //~^^^ ERROR lower range bound must be less than upper

    match 5u64 {
        0xFFFF_FFFF_FFFF_FFFF ..= 1 => { }
        _ => { }
    };
    //~^^^ ERROR lower range bound must be less than or equal to upper
}
