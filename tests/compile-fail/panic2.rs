//error-pattern: the evaluated program panicked

fn main() {
    std::panic!("{}-panicking from libstd", 42);
}
