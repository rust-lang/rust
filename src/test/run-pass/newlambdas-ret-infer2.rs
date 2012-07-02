// xfail-test fn~ is not inferred
// Test that the lambda kind is inferred correctly as a return
// expression

fn shared() -> fn@() { || () }

fn unique() -> fn~() { || () }

fn main() {
}
