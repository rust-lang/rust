#![u=||{static d=||1;}]
//~^ attribute value must be a literal
//~| cannot find attribute `u` in this scope
//~| missing type for `static` item

#![a={impl std::ops::Neg for i8 {}}]
//~^ ERROR attribute value must be a literal
//~| ERROR cannot find attribute `a` in this scope
//~| ERROR `main` function not found in crate `issue_90873`
