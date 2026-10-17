struct Obj<F> where F: FnMut() -> u32 {
    closure: F,
}

fn main() {
    let o = Obj { closure: || 42 };
    o.closure();
    //~^ ERROR no method named `closure` found
}
