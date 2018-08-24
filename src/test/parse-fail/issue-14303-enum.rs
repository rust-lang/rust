// compile-flags: -Z parse-only

enum X<'a, T, 'b> {
//~^ ERROR lifetime parameters must be declared prior to type parameters
    A(&'a T)
}
