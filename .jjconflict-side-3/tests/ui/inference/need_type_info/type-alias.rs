// Test the inference errors in case the relevant path
// uses a type alias.
//
// Regression test for #97698.
struct Ty<T>(T);
impl<T> Ty<T> {
    fn new() {}
}

type DirectAlias<T> = Ty<T>;
fn direct_alias() {
    DirectAlias::new()
    //~^ ERROR type annotations needed
}

type IndirectAlias<T> = Ty<Box<T>>;
fn indirect_alias() {
    IndirectAlias::new(); //~ ERROR: type annotations needed
    // FIXME: This should also emit an error.
    //
    // Added it separately as `type-alias-indirect.rs`
    // where it does error.
}

struct TyDefault<T, U = u32>(T, U);
impl<T> TyDefault<T> {
    fn new() {}
}

type DirectButWithDefaultAlias<T> = TyDefault<T>;
fn direct_but_with_default_alias() {
    DirectButWithDefaultAlias::new();
    //~^ ERROR type annotations needed
}

fn main() {}
