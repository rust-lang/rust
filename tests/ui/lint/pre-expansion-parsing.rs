//@check-pass
 
#![crate_type = "lib"]

#[expect] // OK
#[expect[wut]] // OK
#[expect(expect)] // OK
#[expect(expect(expect))] //OK
#[deny(({!}))] // OK
#[expect[helix::<ub>]] // OK
#[cfg(false)]
const _: () = ();
