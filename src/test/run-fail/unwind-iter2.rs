// error-pattern:fail

iter x() -> int {
    let a = @0;
    put 1;
}

fn main() {
    for each x in x() {
        fail;
    }
}