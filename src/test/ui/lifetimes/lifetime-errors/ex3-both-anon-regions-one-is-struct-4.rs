struct Ref<'a, 'b> { a: &'a u32, b: &'b u32 }

fn foo(mut y: Ref, x: &u32) {
    y.b = x; //~ ERROR lifetime mismatch
}

fn main() { }
