// run-pass
// pretty-expanded FIXME #23616

struct S<T>(#[allow(unused_tuple_struct_fields)] T);

pub fn main() {
    let _s = S(2);
}
