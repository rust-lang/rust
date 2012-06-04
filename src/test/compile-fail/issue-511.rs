use std;
import option;

fn f<T>(&o: option<T>) {
    assert o == option::none;
}

fn main() {
    f::<int>(option::none);
    //!^ ERROR taking mut reference to static item
    //!^^ ERROR illegal borrow unless pure: creating mutable alias to aliasable, immutable memory
    //!^^^ NOTE impure due to access to impure function
}