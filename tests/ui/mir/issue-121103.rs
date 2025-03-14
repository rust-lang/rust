fn main(_: <lib2::GenericType<42> as lib2::TypeFn>::Output) {}
//~^ ERROR failed to resolve: use of unresolved module or unlinked crate `lib2`
//~| ERROR failed to resolve: use of unresolved module or unlinked crate `lib2`
