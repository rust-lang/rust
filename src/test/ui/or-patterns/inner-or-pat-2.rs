#[allow(unused_variables)]
#[allow(unused_parens)]
fn main() {
    let x = "foo";
    match x {
        x @ ((("h" | "ho" | "yo" | ("dude" | "w")) | () | "nop") | ("hey" | "gg")) |
        //~^ ERROR mismatched types
        x @ ("black" | "pink") |
        x @ ("red" | "blue") => {
        }
        _ => (),
    }
}
