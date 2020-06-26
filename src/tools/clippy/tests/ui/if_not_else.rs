#![warn(clippy::all)]
#![warn(clippy::if_not_else)]

fn bla() -> bool {
    unimplemented!()
}

fn main() {
    if !bla() {
        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if 4 != 5 {
        println!("Bugs");
    } else {
        println!("Bunny");
    }
}
