static tmp : [&'static str; 2]  = ["hello", "he"];

fn main() {
    let z = "hello";
    match z {
        tmp[0] => {} //~ ERROR expected one of `=>`, `@`, `if`, or `|`, found `[`
        _ => {}
    }
}
