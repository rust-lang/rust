#![warn(clippy::cyclomatic_complexity)]
#![warn(unused)]

fn main() {
    kaboom();
}

#[clippy::cyclomatic_complexity = "0"]
fn kaboom() {
    if 42 == 43 {
        panic!();
    } else if "cake" == "lie" {
        println!("what?");
    }
}
