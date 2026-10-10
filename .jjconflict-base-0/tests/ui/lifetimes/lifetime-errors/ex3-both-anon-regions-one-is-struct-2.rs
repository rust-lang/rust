struct Ref<'a, 'b> { a: &'a u32, b: &'b u32 }

fn foo(mut x: Ref, y: &u32) {
    y = x.b;
    //~^ ERROR lifetime may not live long enough
    //~| ERROR cannot assign to immutable argument
}

fn main() { }
