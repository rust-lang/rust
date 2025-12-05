// It might be intuitive for a user coming from languages like Java
// to declare a method directly in a struct's definition. Make sure
// rustc can give a helpful suggestion.
// Suggested in issue #76421

struct S {
    field: usize,

    fn foo() {}
    //~^ ERROR functions are not allowed in struct definitions
    //~| HELP unlike in C++, Java, and C#, functions are declared in `impl` blocks
    //~| HELP see https://doc.rust-lang.org/book/ch05-03-method-syntax.html for more information
}

union U {
    variant: usize,

    fn foo() {}
    //~^ ERROR functions are not allowed in union definitions
    //~| HELP unlike in C++, Java, and C#, functions are declared in `impl` blocks
    //~| HELP see https://doc.rust-lang.org/book/ch05-03-method-syntax.html for more information
}

enum E {
    Variant,

    fn foo() {}
    //~^ ERROR functions are not allowed in enum definitions
    //~| HELP unlike in C++, Java, and C#, functions are declared in `impl` blocks
    //~| HELP see https://doc.rust-lang.org/book/ch05-03-method-syntax.html for more information
    //~| HELP enum variants can be `Variant`, `Variant = <integer>`, `Variant(Type, ..., TypeN)` or `Variant { fields: Types }`
}

fn main() {}
