//@ error-pattern:can't use generic parameters from outer item
fn hd<U>(v: Vec<U> ) -> U {
    fn hd1(w: [U]) -> U { return w[0]; }
    //~^ ERROR can't use generic parameters from outer item
    //~| ERROR can't use generic parameters from outer item

    return hd1(v);
}

fn main() {}
