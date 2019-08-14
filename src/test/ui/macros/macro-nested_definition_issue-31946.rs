// run-pass
fn main() {
    println!("{}", {
        macro_rules! foo {
            ($name:expr) => { concat!("hello ", $name) }
        }
        foo!("rust")
    });
}
