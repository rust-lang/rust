struct Ref<'a> {
    x: &'a u32,
}

fn foo(mut x: Vec<Ref>, y: Ref) {
    x.push(y);
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
