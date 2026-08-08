//@ known-bug: rust-lang/rust#146903
#![feature(generic_const_exprs)]
#![feature(lazy_type_alias)]
type FooArg<'a, 'bb> = [u8; x];
type _TaWhere1 = Box<Fn(FooArg)>;
