// error-pattern:Unsatisfied precondition constraint
pure fn even(x: uint) -> bool {
    if x < 2u {
        ret false;
    } else if x == 2u { ret true; } else { ret even(x - 2u); }
}

fn print_even(x: uint) : even(x) { log x; }

fn foo(x: uint) { if check even(x) { fail; } else { print_even(x); } }

fn main() { foo(3u); }
