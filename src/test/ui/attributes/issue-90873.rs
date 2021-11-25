#![u=||{static d=||1;}]
//~^ unexpected token
//~| cannot find attribute `u` in this scope
//~| `main` function not found in crate `issue_90873`
//~| missing type for `static` item

#![a={impl std::ops::Neg for i8 {}}]
//~^ ERROR unexpected token
//~| ERROR cannot find attribute `a` in this scope
