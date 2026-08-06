#![feature(f128)]
#![feature(f16)]

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

fn main() {
    m!(0, ..u8::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..u16::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..u32::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..u64::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..u128::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum

    m!(0, ..i8::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..i16::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..i32::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..i64::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0, ..i128::MIN);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum

    m!(0f16, ..f16::NEG_INFINITY);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0f32, ..f32::NEG_INFINITY);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0f64, ..f64::NEG_INFINITY);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
    m!(0f128, ..f128::NEG_INFINITY);
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum

    m!('a', ..'\u{0}');
    //~^ ERROR exclusive upper bound for a range bound cannot be the minimum
}
