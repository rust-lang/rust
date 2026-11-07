//@ run-rustfix

fn foo<N>(_x: N) {}
//~^ NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here
//~| NOTE function defined here

fn main() {
    foo::<u64>(42_usize);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_u64);
    foo::<u64>(42_u32);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_u16);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_u8);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_isize);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_i64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_i32);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_i16);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42_i8);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42.0_f64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<u64>(42.0_f32);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
}
