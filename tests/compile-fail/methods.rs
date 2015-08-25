#![feature(plugin)]
#![plugin(clippy)]

#[deny(option_unwrap_used, result_unwrap_used)]
#[deny(str_to_string, string_to_string)]
fn main() {
    let opt = Some(0);
    let _ = opt.unwrap();  //~ERROR used unwrap() on an Option

    let res: Result<i32, ()> = Ok(0);
    let _ = res.unwrap();  //~ERROR used unwrap() on a Result

    let _ = "str".to_string();  //~ERROR `"str".to_owned()` is faster

    let v = &"str";
    let string = v.to_string();  //~ERROR `(*v).to_owned()` is faster
    let _again = string.to_string();  //~ERROR `String.to_string()` is a no-op
}
