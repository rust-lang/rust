// error-pattern:fail

iter x() -> int {
    fail;
    put 0;
}

fn main() {
    let a = @0;
    for each x in x() {
    }
}