fn f() -> impl Sized {
    2.0E
    //~^ ERROR expected at least one digit in exponent
}

fn main() {}
