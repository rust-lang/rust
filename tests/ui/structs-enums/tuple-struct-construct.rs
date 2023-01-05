// run-pass
#[allow(unused_tuple_struct_fields)]
#[derive(Debug)]
struct Foo(isize, isize);

pub fn main() {
    let x = Foo(1, 2);
    println!("{:?}", x);
}
