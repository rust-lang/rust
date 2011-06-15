

fn leaky[T](&T t) { }

fn main() { auto x = @10; leaky[@int](x); }