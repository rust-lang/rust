#![warn(clippy::map_flatten)]
#![feature(result_flattening)]
//@no-rustfix
// issue #8506, multi-line
#[rustfmt::skip]
fn long_span() {
    let _: Option<i32> = Some(1)
        .map(|x| {
        //~^ ERROR: called `map(..).flatten()` on `Option`
        //~| NOTE: `-D clippy::map-flatten` implied by `-D warnings`
            if x <= 5 {
                Some(x)
            } else {
                None
            }
        })
        .flatten();

    let _: Result<i32, i32> = Ok(1)
        .map(|x| {
        //~^ ERROR: called `map(..).flatten()` on `Result`
            if x == 1 {
                Ok(x)
            } else {
                Err(0)
            }
        })
        .flatten();

    let result: Result<i32, i32> = Ok(2);
    fn do_something() { }
    let _: Result<i32, i32> = result
        .map(|res| {
        //~^ ERROR: called `map(..).flatten()` on `Result`
            if res > 0 {
                do_something();
                Ok(res)
            } else {
                Err(0)
            }
        })
        .flatten();
        
    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .map(|some_value| {
        //~^ ERROR: called `map(..).flatten()` on `Iterator`
            if some_value > 3 {
                Some(some_value)
            } else {
                None
            }
        })
        .flatten()
        .collect();
}

fn main() {
    long_span();
}
