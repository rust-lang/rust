static tmp : [&'static str; 2]  = ["hello", "he"];

fn main() {
    let z = "hello";
    match z {
        tmp[0] => {} //~ error: expected a pattern, found an expression
        _ => {}
    }
}
