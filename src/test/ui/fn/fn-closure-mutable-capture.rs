pub fn bar<F: Fn()>(_f: F) {}

pub fn foo() {
    let mut x = 0;
    bar(move || x = 1);
    //~^ ERROR cannot assign to captured outer variable in an `Fn` closure
    //~| NOTE `Fn` closures cannot capture their enclosing environment for modifications
}

fn main() {}
