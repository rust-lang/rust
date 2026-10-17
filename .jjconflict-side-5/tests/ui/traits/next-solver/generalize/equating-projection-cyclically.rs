//@ check-pass
//@ compile-flags: -Znext-solver

trait Test {
    type Assoc;
}

fn transform<T: Test>(x: Inv<T>) -> Inv<T::Assoc> {
    todo!()
}

impl Test for i32 {
    type Assoc = i32;
}

impl Test for String {
    type Assoc = String;
}

struct Inv<T>(Option<*mut T>);

fn main() {
    let mut x: Inv<_> = Inv(None);
    // This ends up equating `Inv<?x>` with `Inv<<?x as Test>::Assoc>`
    // which fails the occurs check when generalizing `?x`.
    //
    // We end up emitting a delayed obligation, causing this to still
    // succeed.
    x = transform(x);
    x = Inv::<i32>(None);
}
