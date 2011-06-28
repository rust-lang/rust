// error-pattern:Copying a non-copyable type

resource foo(int i) {}

fn main() {
    auto x <- foo(10);
    auto y = x;
}
