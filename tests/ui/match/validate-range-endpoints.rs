#![allow(overlapping_range_endpoints)]

fn main() {
    const TOO_BIG: u8 = 256;
    match 0u8 {
        1..257 => {}
        //~^ ERROR literal out of range
        1..=256 => {}
        //~^ ERROR literal out of range

        // overflow is detected in a later pass for these
        0..257 => {}
        0..=256 => {}
        256..=100 => {}

        // There isn't really a way to detect these
        1..=TOO_BIG => {}
        //~^ ERROR lower range bound must be less than or equal to upper
        _ => {}
    }

    match 0u64 {
        10000000000000000000..=99999999999999999999 => {}
        //~^ ERROR literal out of range
        _ => {}
    }

    match 0i8 {
        0..129 => {}
        //~^ ERROR literal out of range
        0..=128 => {}
        //~^ ERROR literal out of range
        -129..0 => {}
        //~^ ERROR literal out of range
        -10000..=-20 => {}
        //~^ ERROR literal out of range

        // overflow is detected in a later pass for these
        128..=0 => {}
        0..-129 => {}
        -10000..=0 => {}
        _ => {}
    }

    // FIXME: error message is confusing
    match 0i8 {
        //~^ ERROR `i8::MIN..=-17_i8` and `1_i8..=i8::MAX` not covered
        -10000..=0 => {}
    }
    match 0i8 {
        //~^ ERROR `i8::MIN..=-17_i8` not covered
        -10000.. => {}
    }
}
