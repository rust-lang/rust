macro_rules! m {
    () => {
        {}
    };
}

enum Enum { A, B }

fn main() {
    match Enum::A {
    //~^ ERROR non-exhaustive patterns
    Enum::A => m!()
    }
}
