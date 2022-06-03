// run-rustfix

fn foo<N>(_x: N) {}

fn main() {
    foo::<u32>(42_usize);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_u64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_u32);
    foo::<u32>(42_u16);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_u8);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_isize);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_i64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_i32);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_i16);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42_i8);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42.0_f64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    foo::<u32>(42.0_f32);
    //~^ ERROR mismatched types
    //~| NOTE expected
}
