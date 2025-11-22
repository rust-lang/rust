//@ run-pass
fn main() {
    let float = 0.123_456_789;
    // this special case is exempted from warning
    println!("{:.}", float);
    println!("Missing precision: {float:.} {float:3.} {float:_^12.}");
    //~^ WARNING invalid format string: expected numerical precision after precision specifier
    //~| WARNING invalid format string: expected numerical precision after precision specifier
    //~| WARNING invalid format string: expected numerical precision after precision specifier
    println!("Given precision: {float:.6} {float:3.15} {float:_^12.7}");
}
