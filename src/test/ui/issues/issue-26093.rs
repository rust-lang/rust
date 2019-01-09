macro_rules! not_a_place {
    ($thing:expr) => {
        $thing = 42;
        //~^ ERROR invalid left-hand side expression
    }
}

fn main() {
    not_a_place!(99);
}
