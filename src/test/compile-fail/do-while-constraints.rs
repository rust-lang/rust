// error-pattern:unsatisfied precondition constraint (for example, init(y
fn main() {

    let y: int = 42;
    let x: int;
    loop {
        log(debug, y);
        do  { do  { do  { x <- y; } while true } while true } while true
    }
}
