// run-rustfix

fn main() {
    println!();
    println!("");

    match "a" {
        _ => println!(""),
    }
}
