fn take_str_maybe(x: Option<&str>) -> Option<&str> { None }

fn main() {
    let string = String::from("Hello, world");
    let option = Some(&string);
    take_str_maybe(option);
    //~^ ERROR: mismatched types [E0308]
}
