fn foo<U>(t: U) {
    let y = t();
//~^ ERROR: expected function, found `U`
}

struct Bar;

pub fn some_func() {
    let f = Bar();
//~^ ERROR: expected function, found `Bar`
}

fn main() {
    foo(|| { 1 });
}
