// error-pattern: Unsatisfied precondition constraint (for example, init(y
fn main() {

    let y: int = 42;
    let x: int;
    do  {
        log_full(core::debug, y);
        do  { do  { do  { x <- y; } while true } while true } while true
    } while true
}
