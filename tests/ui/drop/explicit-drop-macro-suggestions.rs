//@ run-rustfix
// https://github.com/rust-lang/rust/issues/164044
macro_rules! generated {
    () => {
        Drop::drop(&mut vec![3_u8]); //~ ERROR explicit use of destructor method
        //~| HELP consider using `drop` function
    };
}

fn main() {
    vec![1_u8].drop(); //~ ERROR explicit use of destructor method
    //~| HELP consider using `drop` function

    Drop::drop(&mut vec![2_u8]); //~ ERROR explicit use of destructor method
    //~| HELP consider using `drop` function

    generated!();
}
