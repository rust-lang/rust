#![feature(trait_alias, const_trait_impl)]

const trait Bar {}

const trait Foo = Bar;

const impl Bar for () {}

struct X;

const impl X {}
