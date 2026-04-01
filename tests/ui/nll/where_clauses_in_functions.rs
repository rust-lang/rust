#![allow(dead_code)]

fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> (&'a u32, &'b u32)
where
    'a: 'b,
{
    (x, y)
}

fn bar<'a, 'b>(x: &'a u32, y: &'b u32) -> (&'a u32, &'b u32) {
    foo(x, y)
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
