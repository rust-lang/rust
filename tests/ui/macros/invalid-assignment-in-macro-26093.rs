// https://github.com/rust-lang/rust/issues/26093
macro_rules! not_a_place {
    ($thing:expr) => {
        $thing = 42;
        //~^ ERROR invalid left-hand side of assignment
        $thing += 42;
        //~^ ERROR invalid left-hand side of assignment
    }
}

fn main() {
    not_a_place!(99);
}
