// Regression test for issue #97226

fn test_fn() -> impl ?Sized {} //~ ERROR return type should be sized

fn main() {}
