#![crate_type = "lib"]
#![feature(extern_item_impls)]
// `eii` expands to, among other things, `macro eii() {}`.
// If we have two eiis named the same thing, we have a duplicate definition
// for that macro. The compiler happily continues compiling on duplicate
// definitions though, to emit as many diagnostics as possible.
// However, in the case of eiis, this can break the assumption that every
// eii has only one default implementation, since the default for both eiis will
// name resolve to the same eii definiton (since the other definition was duplicate)
// This test tests for the previously-ICE that occurred when this assumption
// (of 1 default) was broken which was reported in #149982.

#[eii(eii1)]
fn a() {}

#[eii(eii1)]
//~^ ERROR the name `eii1` is defined multiple times
fn b() {}
