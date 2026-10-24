// Test that we don't assume that the penultimate segment of an enum variant path refers to the
// enum just because it has generic arguments.
//
// We once used to accept the code below interpreting `module::<i32>::Variant`
// as `module::Variant::<i32>` basically.
//
// issue: <https://github.com/rust-lang/rust/issues/154962>

enum Enum<T> { Variant, Carry(T) }

mod module { pub(super) use super::Enum::Variant; }

fn main() {
    let _ = module::<i32>::Variant;
    //~^ ERROR type arguments are not allowed on module `module`

    let _ = self::module::<()>::Variant {};
    //~^ ERROR type arguments are not allowed on module `module`
}
