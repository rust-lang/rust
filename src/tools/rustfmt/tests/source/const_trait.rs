#![feature(trait_alias, const_trait_impl)]

const trait Bar {}

const trait Foo = Bar;

impl const Bar for () {}

// const impl gets reformatted to impl const.. for now
const impl Bar for u8 {}

struct X;

const impl X {}
