#![warn(clippy::implicit_clone, clippy::string_to_string)]
#![allow(clippy::redundant_clone, clippy::unnecessary_literal_unwrap)]

fn main() {
    let mut message = String::from("Hello");
    let mut v = message.to_string();
    //~^ ERROR: implicitly cloning a `String` by calling `to_string` on its dereferenced type

    let variable1 = String::new();
    let v = &variable1;
    let variable2 = Some(v);
    let _ = variable2.map(|x| {
        println!();
        x.to_string()
        //~^ ERROR: implicitly cloning a `String` by calling `to_string` on its dereferenced type
    });

    let x = Some(String::new());
    let _ = x.unwrap_or_else(|| v.to_string());
    //~^ ERROR: implicitly cloning a `String` by calling `to_string` on its dereferenced type
}
