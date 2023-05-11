// compile-flags: -Ztrait-solver=next
// known-bug: unknown

trait Test {
    type Assoc;
}

fn transform<T: Test>(x: T) -> T::Assoc {
    todo!()
}

impl Test for i32 {
    type Assoc = i32;
}

impl Test for String {
    type Assoc = String;
}

fn main() {
    let mut x = Default::default();
    x = transform(x);
    x = 1i32;
}
