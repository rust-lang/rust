#![crate_type = "lib"]

#[expect] // OK
#[expect[wut]] // OK
#[expect(expect)] // OK
#[expect(expect(expect))] //~ ERROR malformed
#[deny(({!}))] // OK
#[expect[helix::<ub>]] // OK
#[allow = 1u7] // OK
#[allow = "x"suffix] // OK
#[allow(foo(a b))] // OK
#[cfg(false)]
const _: () = ();
