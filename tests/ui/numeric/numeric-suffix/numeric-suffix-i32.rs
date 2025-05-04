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
    foo::<i32>(42_usize);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_u64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_u32);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_u16);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_u8);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_isize);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_i64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_i32);
    foo::<i32>(42_i16);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42_i8);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42.0_f64);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
    foo::<i32>(42.0_f32);
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE arguments
}
