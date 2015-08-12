#![feature(plugin)]
#![plugin(clippy)]

#[deny(option_unwrap_used, result_unwrap_used)]
#[deny(str_to_string, string_to_string)]
fn main() {
    let opt = Some(0);
    let _ = opt.unwrap();  //~ERROR

    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();  //~ERROR

    let string = "str".to_string();  //~ERROR
    let _again = string.to_string();  //~ERROR
}
