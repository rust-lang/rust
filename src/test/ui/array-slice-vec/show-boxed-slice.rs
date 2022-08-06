// run-pass

#[derive(Debug)]
struct Foo(#[allow(unused_tuple_struct_fields)] Box<[u8]>);

pub fn main() {
    println!("{:?}", Foo(Box::new([0, 1, 2])));
}
