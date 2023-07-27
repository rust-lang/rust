#![warn(clippy::map_flatten)]
#![feature(result_flattening)]
//@no-rustfix
// issue #8506, multi-line
#[rustfmt::skip]
fn long_span() {
    let _: Option<i32> = Some(1)
        .map(|x| {
            if x <= 5 {
                Some(x)
            } else {
                None
            }
        })
        .flatten();

    let _: Result<i32, i32> = Ok(1)
        .map(|x| {
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
