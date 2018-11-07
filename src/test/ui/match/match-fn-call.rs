use std::path::Path;

fn main() {
    let path = Path::new("foo");
    match path {
        Path::new("foo") => println!("foo"),
        //~^ ERROR expected tuple struct/variant
        Path::new("bar") => println!("bar"),
        //~^ ERROR expected tuple struct/variant
        _ => (),
    }
}
