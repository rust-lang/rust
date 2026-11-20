#![allow(dead_code)]

#[repr(C, packed)]
struct Foo<'a> {
    val: Bar<'a>,
}

struct Bar<'a>(&'a mut i32, Box<()>);

fn main() {
    let _v;
    {
        let mut b = Box::new(1);
        _v = Foo {
            val: Bar(&mut b, Box::new(())),
        };
    }
}
