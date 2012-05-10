// error-pattern:unsatisfied precondition constraint (for example, init(y
fn main() {

    let y: int = 42;
    let x: int;
    loop {
        log(debug, y);
        while true { while true { while true { x <- y; } } }
    }
}
