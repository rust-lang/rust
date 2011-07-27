// xfail-stage0
// error-pattern: Unsatisfied precondition constraint (for example, even(y

fn print_even(y: int) : even(y) { log y; }

pred even(y: int) -> bool { true }

fn main() {

    let y: int = 42;
    let x: int = 1;
    check (even(y));
    while true {
        print_even(y);
        while true { while true { while true { y += x; } } }
    }
}