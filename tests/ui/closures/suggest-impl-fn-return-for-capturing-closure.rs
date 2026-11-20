//! Regression test for <https://github.com/rust-lang/rust/issues/159481>.
//! When a capturing closure is returned where a `fn(...)` pointer return type is
//! expected, suggest changing the return type to `impl Fn(...)`.

#![crate_type = "lib"]

fn no_args(a: i32) -> fn() -> i32 {
    || a
    //~^ ERROR mismatched types
}

fn no_args_no_return(a: i32) -> fn() {
    || println!("{}", a)
    //~^ ERROR mismatched types
}

fn one_arg(a: i32) -> fn(i32) -> i32 {
    |x| x + a
    //~^ ERROR mismatched types
}

fn one_arg_unsafe(a: i32) -> unsafe fn(i32) -> i32 {
    |x| x + a
    //~^ ERROR mismatched types
}

fn one_arg_extern(a: i32) -> extern "C" fn(i32) -> i32 {
    |x| x + a
    //~^ ERROR mismatched types
}

fn one_arg_no_return(a: i32) -> fn(i32) {
    |x| println!("{}", x + a)
    //~^ ERROR mismatched types
}

fn one_arg_return_unit(a: i32) -> fn(i32) -> () {
    |x| println!("{}", x + a)
    //~^ ERROR mismatched types
}

fn one_arg_ref(a: i32) -> fn(&i32) -> &i32 {
    |x| {
        //~^ ERROR mismatched types
        println!("{}", a);
        x
    }
}

fn multi_args(a: i32, b: i32) -> fn(i32, i32) -> i32 {
    |x, y| x + y + a + b
    //~^ ERROR mismatched types
}

fn hrtb(x: i32) -> for<'a> fn(&'a i32) -> &'a i32 {
    |y| &(x + y)
    //~^ ERROR mismatched types
}

fn hrtb_out_of_order(x: i32) -> for<'b, 'a> fn(&'a i32, &'b i32) -> (&'b i32, &'a i32) {
    |y, z| (&(x + y + z), unimplemented!())
    //~^ ERROR mismatched types
}
