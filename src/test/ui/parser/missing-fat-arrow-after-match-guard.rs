fn main() {
    let x = 1;
    let y = 2;
    let value = 3;

    match value {
        Some(x) if x == y {
            //~^ ERROR expected one of `!`, `.`, `::`, `=>`, `?`, or an operator, found `{`
            x
        },
        _ => {
            y
        }
    }
}
