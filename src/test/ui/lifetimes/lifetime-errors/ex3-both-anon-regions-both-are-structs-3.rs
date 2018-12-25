struct Ref<'a, 'b> {
    a: &'a u32,
    b: &'b u32,
}

fn foo(mut x: Ref) {
    x.a = x.b; //~ ERROR lifetime mismatch
}

fn main() {}
