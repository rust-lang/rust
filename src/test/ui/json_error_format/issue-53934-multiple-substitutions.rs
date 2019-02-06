// compile-flags: --error-format pretty-json -Zunstable-options

fn main() {
    let xs = vec![String::from("foo")];
    let d: &Display = &xs;
    println!("{}", d);
}
