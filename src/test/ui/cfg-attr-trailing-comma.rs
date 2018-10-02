// compile-flags: --cfg TRUE

#[cfg_attr(TRUE, inline,)] // OK
fn f() {}

#[cfg_attr(FALSE, inline,)] // OK
fn g() {}

#[cfg_attr(TRUE, inline,,)] //~ ERROR expected `)`, found `,`
fn h() {}

#[cfg_attr(FALSE, inline,,)] //~ ERROR expected `)`, found `,`
fn i() {}
