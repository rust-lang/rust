#![feature(plugin)]
#![plugin(clippy)]

#[deny(option_unwrap_used, result_unwrap_used)]
fn main() {
    let opt = Some(0);
    let _ = opt.unwrap();  //~ERROR

    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();  //~ERROR
}
