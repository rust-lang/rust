#![crate_type = "lib"]

struct Apple((Apple, Option(Banana ? Citron)));
//~^ ERROR invalid `?` in type
//~| ERROR unexpected token: `Citron`
