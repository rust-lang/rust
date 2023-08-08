// compile-flags: -Ztrait-solver=next
// check-pass

fn needs_send(_: impl Send) {}

fn main() {
    needs_send(1);
}
