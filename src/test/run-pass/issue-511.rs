use std;
import std::option;

fn f<@T>(&o: mutable option::t<T>) {
    assert o == option::none;
}

fn main() {
    f::<int>(option::none);
}