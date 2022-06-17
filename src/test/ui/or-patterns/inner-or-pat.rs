// run-pass

#[allow(unused_variables)]
#[allow(unused_parens)]
fn main() {
    let x = "foo";
    match x {
        x @ ((("h" | "ho" | "yo" | ("dude" | "w")) | "no" | "nop") | ("hey" | "gg")) |
        x @ ("black" | "pink") |
        x @ ("red" | "blue") => {
        }
        _ => (),
    }
}
