use std;
import option;

fn f<T>(&o: option<T>) {
    assert o == option::none;
}

fn main() {
    f::<int>(option::none);
}