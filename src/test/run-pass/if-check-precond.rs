pred even(x: uint) -> bool {
    if x < 2u {
        ret false;
    } else if (x == 2u) { ret true; } else { ret even(x - 2u); }
}

fn print_even(x: uint) : even(x) { log x; }

fn foo(x: uint) { if check even(x) { print_even(x); } else { fail; } }

fn main() { foo(2u); }