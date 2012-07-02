// Test that the lambda kind is inferred correctly as a return
// expression

fn shared() -> fn@() { ret || (); }

fn unique() -> fn~() { ret || (); }

fn main() {
}
