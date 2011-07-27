// xfail-stage0
// error-pattern: Unsatisfied precondition constraint (for example, even(y

fn print_even(y: int) { log y; }

pred even(y: int) -> bool { true }

fn main() {

    let y: int = 42;
    check (even(y));
    do  {
        print_even(y);
        do  { do  { do  { y += 1; } while true } while true } while true
    } while true
}