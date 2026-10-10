//@ compile-flags: -Znext-solver
//@ check-pass

fn needs_send(_: impl Send) {}

fn main() {
    needs_send(1);
}
