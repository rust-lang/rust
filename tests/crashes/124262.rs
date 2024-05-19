//@ known-bug: #124262
//@ edition:2021

struct Foo(<&[fn()] as ::core::ops::Deref>::Target);
const _: *const Foo = 0 as _;
