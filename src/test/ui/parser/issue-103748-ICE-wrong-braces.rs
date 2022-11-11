#![crate_type = "lib"]

struct Apple((Apple, Option(Banana ? Citron)));
//~^ ERROR invalid `?` in type
//~| ERROR expected one of `)` or `,`, found `Citron`
//~| ERROR cannot find type `Citron` in this scope [E0412]
//~| ERROR parenthesized type parameters may only be used with a `Fn` trait [E0214]
//~| ERROR recursive type `Apple` has infinite size [E0072]
