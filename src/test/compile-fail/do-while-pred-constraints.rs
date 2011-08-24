// error-pattern: Unsatisfied precondition constraint (for example, even(y

fn print_even(y: int) : even(y) { log y; }

pure fn even(y: int) -> bool { true }

fn main() {
    let y: int = 42;
    check (even(y));
    do  {
        print_even(y);
        do  { do  { do  { y += 1; } while true } while true } while true
    } while true
}
