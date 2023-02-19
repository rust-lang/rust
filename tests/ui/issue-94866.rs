macro_rules! m {
    () => {
        {}
    };
}

enum Enum { A, B }

fn main() {
    match Enum::A {
    //~^ ERROR match is non-exhaustive
    Enum::A => m!()
    }
}
