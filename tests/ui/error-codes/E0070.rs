const SOME_CONST : i32 = 12;

fn some_other_func() {}

fn some_function() {
    SOME_CONST = 14; //~ ERROR E0070
    1 = 3; //~ ERROR E0070
    some_other_func() = 4; //~ ERROR E0070
}

fn main() {
}
