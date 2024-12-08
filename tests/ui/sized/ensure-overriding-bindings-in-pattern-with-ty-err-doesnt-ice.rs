fn main() {
    let str::<{fn str() { let str::T>>::as_bytes; }}, T>::as_bytes;
//~^ ERROR expected a pattern, found an expression
//~| ERROR cannot find type `T` in this scope
//~| ERROR const and type arguments are not allowed on builtin type `str`
//~| ERROR expected unit struct, unit variant or constant, found associated function `str<
//~| ERROR type annotations needed
}
