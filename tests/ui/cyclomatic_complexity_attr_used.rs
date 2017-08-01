#![feature(plugin, custom_attribute)]
#![plugin(clippy)]
#![warn(cyclomatic_complexity)]
#![warn(unused)]

fn main() {
    kaboom();
}

#[cyclomatic_complexity = "0"]
fn kaboom() {
    if 42 == 43 {
        panic!();
    } else if "cake" == "lie" {
        println!("what?");
    }
}
