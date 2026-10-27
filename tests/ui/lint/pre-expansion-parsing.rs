#![crate_type = "lib"]

#[expect] // OK
#[expect[wut]] // OK
#[expect(expect)] // OK
#[expect(expect(expect))] //~ ERROR expected a lint
#[deny(({!}))] // OK
#[expect[helix::<ub>]] // OK
#[cfg(false)]
const _: () = ();
