// error-pattern:duplicate meta item `name`

extern mod std(name = "std", name = "nonstd");

fn main() { }
