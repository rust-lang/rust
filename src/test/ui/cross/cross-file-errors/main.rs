#[macro_use]
mod underscore;

fn main() {
    underscore!();
    //~^ ERROR expected expression, found reserved identifier `_`
    //~^^ ERROR destructuring assignments are unstable
}
