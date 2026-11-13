#![crate_type = "lib"]

#[expect] // OK
#[expect[wut]] // OK
#[expect(expect)] // OK
#[expect(expect(expect))] //~ ERROR malformed lint attribute input
                          //~| ERROR malformed lint attribute input
#[deny(({!}))] // OK
#[expect[helix::<ub>]] // OK
#[cfg(false)]
const _: () = ();
