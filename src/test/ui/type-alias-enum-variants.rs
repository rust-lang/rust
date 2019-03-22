#![feature(type_alias_enum_variants)]

type Alias<T> = Option<T>;

fn main() {
    let _ = Option::<u8>::None; // OK
    let _ = Option::None::<u8>; // OK (Lint in future!)
    let _ = Alias::<u8>::None; // OK
    let _ = Alias::None::<u8>; // Error
    //~^ type arguments are not allowed for this type
}
