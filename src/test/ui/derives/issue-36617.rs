#![derive(Copy)] //~ ERROR cannot determine resolution for the attribute macro `derive`
//~^ ERROR `derive` attribute cannot be used at crate level

#![test]//~ ERROR cannot determine resolution for the attribute macro `test`
//~^ ERROR `test` attribute cannot be used at crate level

#![test_case]//~ ERROR cannot determine resolution for the attribute macro `test_case`
//~^ ERROR `test_case` attribute cannot be used at crate level

#![bench]//~ ERROR cannot determine resolution for the attribute macro `bench`
//~^ ERROR `bench` attribute cannot be used at crate level

#![global_allocator]//~ ERROR cannot determine resolution for the attribute macro `global_allocator`
//~^ ERROR `global_allocator` attribute cannot be used at crate level

fn main() {}
