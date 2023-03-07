// run-pass
// pretty-expanded FIXME #23616

#[derive(Clone)]
#[allow(unused_tuple_struct_fields)]
struct S<T>(T, ());

pub fn main() {
    let _ = S(1, ()).clone();
}
