// error-pattern:Unsatisfied precondition

fn main() {
    obj foo(x: int) {drop { let baz: int; log baz; } }
    fail;
}