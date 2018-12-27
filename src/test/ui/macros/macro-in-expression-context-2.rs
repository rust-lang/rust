macro_rules! empty { () => () }

fn main() {
    match 42 {
        _ => { empty!() }
//~^ ERROR macro expansion ends with an incomplete expression
    };
}
