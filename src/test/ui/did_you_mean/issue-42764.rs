enum DoubleOption<T> {
    FirstSome(T),
    AlternativeSome(T),
    Nothing,
}

fn this_function_expects_a_double_option<T>(d: DoubleOption<T>) {}

fn main() {
    let n: usize = 42;
    this_function_expects_a_double_option(n);
    //~^ ERROR mismatched types
}
