// run-pass
// pretty-expanded FIXME #23616

struct A(#[allow(unused_tuple_struct_fields)] bool);

pub fn main() {
    let f = A;
    f(true);
}
