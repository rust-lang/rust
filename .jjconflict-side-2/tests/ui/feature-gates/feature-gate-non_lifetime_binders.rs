fn foo() where for<T> T:, {}
//~^ ERROR only lifetime parameters can be used in this context

fn main() {}
