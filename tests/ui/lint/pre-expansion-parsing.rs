#![crate_type = "lib"]

#[expect] // OK
#[expect[wut]] // OK
#[expect(expect)] // OK
#[expect(expect(expect))] //~ ERROR malformed
#[deny(({!}))] // OK
#[expect[helix::<ub>]] // OK
#[cfg(false)]
const _: () = ();
