//@ edition: 2021
//@ compile-flags: -Cinstrument-coverage=on

#[inline]
pub fn inline_me() {}

#[inline(never)]
pub fn no_inlining_please() {}

pub fn generic<T>() {}

// FIXME(#132436): Even though this doesn't ICE, it still produces coverage
// reports that undercount the affected code.
