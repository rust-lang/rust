struct X;

impl std::fmt::Display for X {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x")
    }
}

fn main() {
    let x = X;
    println!("test: {x} {x:?}");
    //~^ ERROR: `X` doesn't implement `Debug`
}
