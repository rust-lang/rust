pub fn bar<F: Fn()>(_f: F) {}

pub fn foo() {
    let mut x = 0;
    bar(move || x = 1);
    //~^ ERROR cannot assign to `x`, as it is a captured variable in a `Fn` closure
    //~| NOTE cannot assign
    //~| HELP consider changing this to accept closures that implement `FnMut`
}

fn main() {}
