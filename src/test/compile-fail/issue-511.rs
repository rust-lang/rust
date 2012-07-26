use std;
import option;

fn f<T>(&o: option<T>) {
    assert o == option::none;
}

fn main() {
    f::<int>(option::none);
    //~^ ERROR taking mut reference to static item
    //~^^ ERROR illegal borrow: creating mutable alias to aliasable, immutable memory
}