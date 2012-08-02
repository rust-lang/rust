// Test that the lambda kind is inferred correctly as a return
// expression

fn shared() -> fn@() { return || (); }

fn unique() -> fn~() { return || (); }

fn main() {
}
