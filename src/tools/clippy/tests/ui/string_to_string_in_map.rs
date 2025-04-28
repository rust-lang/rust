#![deny(clippy::string_to_string)]
#![allow(clippy::unnecessary_literal_unwrap, clippy::useless_vec, clippy::iter_cloned_collect)]

fn main() {
    let variable1 = String::new();
    let v = &variable1;
    let variable2 = Some(v);
    let _ = variable2.map(String::to_string);
    //~^ string_to_string
    let _ = variable2.map(|x| x.to_string());
    //~^ string_to_string
    #[rustfmt::skip]
    let _ = variable2.map(|x| { x.to_string() });
    //~^ string_to_string

    let _ = vec![String::new()].iter().map(String::to_string).collect::<Vec<_>>();
    //~^ string_to_string
    let _ = vec![String::new()].iter().map(|x| x.to_string()).collect::<Vec<_>>();
    //~^ string_to_string
}
