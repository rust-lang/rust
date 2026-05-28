//@ known-bug: #136416
#![feature(generic_const_exprs)]
struct State<const S : usize = {}> where[(); S] :;

struct Foo;
struct State2<const S: usize = Foo> where [(); S]:;
