//@ run-rustfix

fn main() {
    let a = Some(42);
    println!(
        "The value is {}.",
        (a.unwrap)() //~ERROR [E0615]
    );
}
