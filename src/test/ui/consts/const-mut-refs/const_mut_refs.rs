// run-pass

#![feature(const_mut_refs)]

struct Foo {
    x: usize
}

const fn bar(foo: &mut Foo) -> usize {
    foo.x + 1
}

fn main() {
    let _: [(); bar(&mut Foo { x: 0 })] = [(); 1];
}
