// Ambiguity between a `macro_rules` macro and a non-existent import recovered as `Res::Err`

macro_rules! mac { () => () }

mod m {
    use nonexistent_module::mac; //~ ERROR unresolved import `nonexistent_module`

    mac!(); //~ ERROR `mac` is ambiguous
}

fn main() {}
