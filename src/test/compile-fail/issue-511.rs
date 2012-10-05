extern mod std;
use cmp::Eq;

fn f<T:Eq>(o: &mut Option<T>) {
    assert *o == option::None;
}

fn main() {
    f::<int>(&mut option::None);
    //~^ ERROR illegal borrow: creating mutable alias to static item
}
