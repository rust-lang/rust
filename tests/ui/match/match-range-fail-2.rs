fn main() {
    match 5 {
        6 ..= 1 => { }
        //~^ ERROR lower range bound must be less than or equal to upper
        _ => { }
    };

    match 5 {
        0 .. 0 => { }
        //~^ ERROR lower range bound must be less than upper
        _ => { }
    };

    match 5u64 {
        0xFFFF_FFFF_FFFF_FFFF ..= 1 => { }
        //~^ ERROR lower range bound must be less than or equal to upper
        _ => { }
    };
}
