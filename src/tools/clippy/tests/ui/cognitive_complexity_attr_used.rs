#![warn(unused, clippy::cognitive_complexity)]
#![allow(unused_crate_dependencies)]

fn main() {
    kaboom();
}

#[clippy::cognitive_complexity = "0"]
fn kaboom() {
    //~^ cognitive_complexity

    if 42 == 43 {
        panic!();
    } else if "cake" == "lie" {
        println!("what?");
    }
}
