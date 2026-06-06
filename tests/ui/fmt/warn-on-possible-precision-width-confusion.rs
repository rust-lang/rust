//@ run-pass
fn main() {
    let float = 0.123_456_789;
    println!("Missing precision: {float:.} {float:3.} {float:_^12.}");
    //~^ WARNING invalid format string: precision specifier without precision value
    //~| WARNING invalid format string: precision specifier without precision value
    println!("Given precision: {float:.6} {float:3.15} {float:_^12.7}");
}
