//! Regression test for <https://github.com/rust-lang/rust/issues/128381>.

struct Struct {
    field: u32,
}

fn main() {
    let mut some_struct = Some(Struct { field: 42 });
    some_struct.as_mut().inspect(|some_struct| {
        some_struct.field *= 10; //~ ERROR cannot assign to `some_struct.field`, which is behind a `&` reference
        // Users can't change type of `some_struct` param, so above error must not suggest it.
    });
}
