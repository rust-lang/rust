// run-pass

#![feature(const_mut_refs)]

struct Foo {
    x: usize
}

const fn bar(foo: &mut Foo) -> usize {
    let x = &mut foo.x;
    *x = 1;
    *x
}

fn main() {
    let _: [(); bar(&mut Foo { x: 0 })] = [(); 1];
}
