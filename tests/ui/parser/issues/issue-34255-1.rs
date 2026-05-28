enum Test {
    Drill {
        field: i32,
    }
}

fn main() {
    Test::Drill(field: 42);
    //~^ ERROR invalid `struct` delimiters or `fn` call arguments
}
