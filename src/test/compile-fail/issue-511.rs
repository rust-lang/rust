extern mod std;
use cmp::Eq;

fn f<T:Eq>(&o: Option<T>) {
    assert o == option::None;
}

fn main() {
    f::<int>(option::None);
    //~^ ERROR taking mut reference to static item
    //~^^ ERROR illegal borrow: creating mutable alias to aliasable, immutable memory
}
