#![warn(clippy::all)]
#![warn(clippy::if_not_else)]

fn foo() -> bool {
    unimplemented!()
}
fn bla() -> bool {
    unimplemented!()
}

fn main() {
    if !bla() {
        //~^ ERROR: unnecessary boolean `not` operation
        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if 4 != 5 {
        //~^ ERROR: unnecessary `!=` operation
        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if !foo() {
        println!("Foo");
    } else if !bla() {
        println!("Bugs");
    } else {
        println!("Bunny");
    }
}
