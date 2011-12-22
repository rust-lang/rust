// error-pattern: Unsatisfied precondition constraint (for example, init(y
fn main() {

    let y: int = 42;
    let x: int;
    while true {
        log_full(core::debug, y);
        while true {
            while true {
                while true { x <- y; }
            }
        }
    }
}
