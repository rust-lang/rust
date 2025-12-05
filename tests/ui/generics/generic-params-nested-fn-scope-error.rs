//! Test that generic parameters from an outer function are not accessible
//! in nested functions.

fn foo<U>(v: Vec<U>) -> U {
    fn bar(w: [U]) -> U {
        //~^ ERROR can't use generic parameters from outer item
        //~| ERROR can't use generic parameters from outer item
        return w[0];
    }

    return bar(v);
}

fn main() {}
