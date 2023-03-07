#[macro_use]
mod underscore;

fn main() {
    underscore!();
    //~^ ERROR `_` can only be used on the left-hand side of an assignment
}
