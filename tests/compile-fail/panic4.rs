//error-pattern: the evaluated program panicked

fn main() {
    core::panic!("{}-panicking from libcore", 42);
}
