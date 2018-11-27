macro_rules! empty { () => () }

fn main() {
    match 42 {
        _ => { empty!() }
//~^ ERROR expected expression, found `<eof>`
    };
}
