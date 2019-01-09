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
    //~| HELP try using a variant of the expected type
}


// But don't issue the "try using a variant" help if the one-"variant" ADT is
// actually a one-field struct.

struct Payload;

struct Wrapper { payload: Payload }

struct Context { wrapper: Wrapper }

fn overton() {
    let _c = Context { wrapper: Payload{} };
    //~^ ERROR mismatched types
}
