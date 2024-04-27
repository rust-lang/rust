//@ check-fail

fn main() {
    println!("{}\
"); //~^ ERROR: 1 positional argument in format string, but no arguments were given
}
