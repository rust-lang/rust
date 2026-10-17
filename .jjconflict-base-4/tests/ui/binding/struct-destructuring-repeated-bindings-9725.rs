// https://github.com/rust-lang/rust/issues/9725
struct A { foo: isize }

fn main() {
    let A { foo, foo } = A { foo: 3 };
    //~^ ERROR: identifier `foo` is bound more than once in the same pattern
    //~^^ ERROR: field `foo` bound multiple times
}
