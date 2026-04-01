//@ run-pass
pub fn main() {
    let command = "a";

    match command {
        "foo" => println!("foo"),
        _     => println!("{}", command),
    }
}
