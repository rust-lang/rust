//@ run-rustfix

#[derive(Debug)]
struct Foo {
    x: isize
}

impl From<Foo> for isize {
    fn from(val: Foo) -> isize {
        val.x
    }
}

fn main() {
    println!("{}", Foo { x: 1 } as isize); //~ ERROR non-primitive cast: `Foo` as `isize` [E0605]
}
