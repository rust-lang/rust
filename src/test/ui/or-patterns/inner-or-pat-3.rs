// run-pass

#[allow(unreachable_patterns)]
#[allow(unused_variables)]
#[allow(unused_parens)]
fn main() {
    let x = "foo";

    match x {
        x @ ("foo" | "bar") |
        (x @ "red" | (x @ "blue" | x @ "red")) => {
        }
        _ => (),
    }
}
