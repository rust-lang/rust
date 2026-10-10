fn main(_: <lib2::GenericType<42> as lib2::TypeFn>::Output) {}
//~^ ERROR: cannot find
//~| ERROR: cannot find
//~| NOTE: use of unresolved module or unlinked crate `lib2`
//~| NOTE: use of unresolved module or unlinked crate `lib2`
