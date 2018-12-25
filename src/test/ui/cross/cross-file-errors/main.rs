#[macro_use]
mod underscore;

fn main() {
    underscore!();
    //~^ ERROR expected expression, found reserved identifier `_`
}
