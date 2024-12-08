//@ check-pass
//@ compile-flags: -Z validate-mir

struct Foo<'a>(&'a mut u32);

impl<'a> Drop for Foo<'a> {
    fn drop(&mut self) {
        *self.0 = 0;
    }
}

fn and() {
    let mut foo = 0;
    // This used to compile also before the fix
    if true && *Foo(&mut foo).0 == 0 && ({ foo = 0; true}) {}

    // This used to fail before the fix
    if *Foo(&mut foo).0 == 0 && ({ foo = 0; true}) {}

    println!("{foo}");
}

fn or() {
    let mut foo = 0;
    // This used to compile also before the fix
    if false || *Foo(&mut foo).0 == 1 || ({ foo = 0; true}) {}

    // This used to fail before the fix
    if *Foo(&mut foo).0 == 1 || ({ foo = 0; true}) {}

    println!("{foo}");
}

fn main() {
    and();
    or();
}
