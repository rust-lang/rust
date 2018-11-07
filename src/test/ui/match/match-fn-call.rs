use std::path::Path;

fn main() {
    let path = Path::new("foo");
    match path {
        Path::new("foo") => println!("foo"),
        Path::new("bar") => println!("bar"),
        _ => (),
    }
}
