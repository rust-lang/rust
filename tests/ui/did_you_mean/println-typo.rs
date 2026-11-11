// https://internals.rust-lang.org/t/18227

fn main() {
    prinltn!(); //~ ERROR cannot find macro `prinltn` in this scope
    //^ a macro with a similar name exists: `println`
}
