struct Ref<'a, 'b> {
    a: &'a u32,
    b: &'b u32,
}

fn foo(mut x: Ref) {
    x.a = x.b;
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
