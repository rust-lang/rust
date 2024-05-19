#![warn(unused, clippy::cognitive_complexity)]
#![allow(unused_crate_dependencies)]

fn main() {
    kaboom();
}

#[clippy::cognitive_complexity = "0"]
fn kaboom() {
    //~^ ERROR: the function has a cognitive complexity of (3/0)
    if 42 == 43 {
        panic!();
    } else if "cake" == "lie" {
        println!("what?");
    }
}
