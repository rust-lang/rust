// error-pattern:Copying a non-copyable type

res foo(int i) {}

fn main() {
    auto x <- foo(10);
    auto y = x;
}
