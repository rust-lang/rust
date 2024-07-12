// https://internals.rust-lang.org/t/18227

fn main() {
    prinltn!(); //~ ERROR cannot find macro `prinltn`
    //^ a macro with a similar name exists: `println`
}
