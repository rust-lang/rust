// error-pattern: mismatched types

fn add1(int i) -> int { ret i+1; }
fn main() {
    auto f = @add1;
    auto g = bind f(5);
}
