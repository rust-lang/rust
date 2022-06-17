#[allow(unused_variables)]
#[allow(unused_parens)]
fn main() {
    let x = "foo";

    match x {
        x @ ("foo" | "bar") |
        (x @ "red" | (x @ "blue" |  "red")) => {
        //~^ variable `x` is not bound in all patterns
        }
        _ => (),
    }
}
