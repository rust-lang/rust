fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> &'b u32 {
    &*x
}

fn main() { }
