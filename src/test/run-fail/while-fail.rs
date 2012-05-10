// error-pattern:giraffe
fn main() {
    fail { while true { fail "giraffe"}; "clandestine" };
}
