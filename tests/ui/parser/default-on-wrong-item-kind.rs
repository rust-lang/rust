// Test parsing for `default` where it doesn't belong.
// Specifically, we are interested in kinds of items or items in certain contexts.
// Also test item kinds in `extern` blocks and associated contexts which are not allowed there.

fn main() {}

#[cfg(false)]
mod free_items {
    default extern crate foo; //~ ERROR an extern crate cannot be `default`
    default use foo; //~ ERROR a `use` import cannot be `default`
    default static foo: u8; //~ ERROR a static item cannot be `default`
    default const foo: u8;
    default fn foo();
    default mod foo {} //~ ERROR a module cannot be `default`
    default extern "C" {} //~ ERROR an extern block cannot be `default`
    default type foo = u8;
    default enum foo {} //~ ERROR an enum cannot be `default`
    default struct foo {} //~ ERROR a struct cannot be `default`
    default union foo {} //~ ERROR a union cannot be `default`
    default trait foo {} //~ ERROR a trait cannot be `default`
    default trait foo = Ord; //~ ERROR a trait alias cannot be `default`
    default impl foo {} //~ ERROR inherent impls cannot be default
    default!();
    default::foo::bar!();
    default default!(); //~ ERROR an item macro invocation cannot be `default`
    default default::foo::bar!(); //~ ERROR an item macro invocation cannot be `default`
    default macro foo {} //~ ERROR a macro definition cannot be `default`
    default macro_rules! foo {} //~ ERROR a macro definition cannot be `default`
}

#[cfg(false)]
extern "C" {
    default extern crate foo; //~ ERROR an extern crate cannot be `default`
    //~^ ERROR extern crate is not supported in `extern` blocks
    default use foo; //~ ERROR a `use` import cannot be `default`
    //~^ ERROR `use` import is not supported in `extern` blocks
    default static foo: u8; //~ ERROR a static item cannot be `default`
    default const foo: u8;
    //~^ ERROR extern items cannot be `const`
    default fn foo();
    default mod foo {} //~ ERROR a module cannot be `default`
    //~^ ERROR module is not supported in `extern` blocks
    default extern "C" {} //~ ERROR an extern block cannot be `default`
    //~^ ERROR extern block is not supported in `extern` blocks
    default type foo = u8;
    default enum foo {} //~ ERROR an enum cannot be `default`
    //~^ ERROR enum is not supported in `extern` blocks
    default struct foo {} //~ ERROR a struct cannot be `default`
    //~^ ERROR struct is not supported in `extern` blocks
    default union foo {} //~ ERROR a union cannot be `default`
    //~^ ERROR union is not supported in `extern` blocks
    default trait foo {} //~ ERROR a trait cannot be `default`
    //~^ ERROR trait is not supported in `extern` blocks
    default trait foo = Ord; //~ ERROR a trait alias cannot be `default`
    //~^ ERROR trait alias is not supported in `extern` blocks
    default impl foo {} //~ ERROR inherent impls cannot be default
    //~^ ERROR implementation is not supported in `extern` blocks
    default!();
    default::foo::bar!();
    default default!(); //~ ERROR an item macro invocation cannot be `default`
    default default::foo::bar!(); //~ ERROR an item macro invocation cannot be `default`
    default macro foo {} //~ ERROR a macro definition cannot be `default`
    //~^ ERROR macro definition is not supported in `extern` blocks
    default macro_rules! foo {} //~ ERROR a macro definition cannot be `default`
    //~^ ERROR macro definition is not supported in `extern` blocks
}

#[cfg(false)]
impl S {
    default extern crate foo; //~ ERROR an extern crate cannot be `default`
    //~^ ERROR extern crate is not supported in `trait`s or `impl`s
    default use foo; //~ ERROR a `use` import cannot be `default`
    //~^ ERROR `use` import is not supported in `trait`s or `impl`s
    default static foo: u8; //~ ERROR a static item cannot be `default`
    //~^ ERROR associated `static` items are not allowed
    default const foo: u8;
    default fn foo();
    default mod foo {}//~ ERROR a module cannot be `default`
    //~^ ERROR module is not supported in `trait`s or `impl`s
    default extern "C" {} //~ ERROR an extern block cannot be `default`
    //~^ ERROR extern block is not supported in `trait`s or `impl`s
    default type foo = u8;
    default enum foo {} //~ ERROR an enum cannot be `default`
    //~^ ERROR enum is not supported in `trait`s or `impl`s
    default struct foo {} //~ ERROR a struct cannot be `default`
    //~^ ERROR struct is not supported in `trait`s or `impl`s
    default union foo {} //~ ERROR a union cannot be `default`
    //~^ ERROR union is not supported in `trait`s or `impl`s
    default trait foo {} //~ ERROR a trait cannot be `default`
    //~^ ERROR trait is not supported in `trait`s or `impl`s
    default trait foo = Ord; //~ ERROR a trait alias cannot be `default`
    //~^ ERROR trait alias is not supported in `trait`s or `impl`s
    default impl foo {} //~ ERROR inherent impls cannot be default
    //~^ ERROR implementation is not supported in `trait`s or `impl`s
    default!();
    default::foo::bar!();
    default default!(); //~ ERROR an item macro invocation cannot be `default`
    default default::foo::bar!(); //~ ERROR an item macro invocation cannot be `default`
    default macro foo {} //~ ERROR a macro definition cannot be `default`
    //~^ ERROR macro definition is not supported in `trait`s or `impl`s
    default macro_rules! foo {} //~ ERROR a macro definition cannot be `default`
    //~^ ERROR macro definition is not supported in `trait`s or `impl`s
}

#[cfg(false)]
trait T {
    default extern crate foo; //~ ERROR an extern crate cannot be `default`
    //~^ ERROR extern crate is not supported in `trait`s or `impl`s
    default use foo; //~ ERROR a `use` import cannot be `default`
    //~^ ERROR `use` import is not supported in `trait`s or `impl`s
    default static foo: u8; //~ ERROR a static item cannot be `default`
    //~^ ERROR associated `static` items are not allowed
    default const foo: u8;
    default fn foo();
    default mod foo {}//~ ERROR a module cannot be `default`
    //~^ ERROR module is not supported in `trait`s or `impl`s
    default extern "C" {} //~ ERROR an extern block cannot be `default`
    //~^ ERROR extern block is not supported in `trait`s or `impl`s
    default type foo = u8;
    default enum foo {} //~ ERROR an enum cannot be `default`
    //~^ ERROR enum is not supported in `trait`s or `impl`s
    default struct foo {} //~ ERROR a struct cannot be `default`
    //~^ ERROR struct is not supported in `trait`s or `impl`s
    default union foo {} //~ ERROR a union cannot be `default`
    //~^ ERROR union is not supported in `trait`s or `impl`s
    default trait foo {} //~ ERROR a trait cannot be `default`
    //~^ ERROR trait is not supported in `trait`s or `impl`s
    default trait foo = Ord; //~ ERROR a trait alias cannot be `default`
    //~^ ERROR trait alias is not supported in `trait`s or `impl`s
    default impl foo {} //~ ERROR inherent impls cannot be default
    //~^ ERROR implementation is not supported in `trait`s or `impl`s
    default!();
    default::foo::bar!();
    default default!(); //~ ERROR an item macro invocation cannot be `default`
    default default::foo::bar!(); //~ ERROR an item macro invocation cannot be `default`
    default macro foo {} //~ ERROR a macro definition cannot be `default`
    //~^ ERROR macro definition is not supported in `trait`s or `impl`s
    default macro_rules! foo {} //~ ERROR a macro definition cannot be `default`
    //~^ ERROR macro definition is not supported in `trait`s or `impl`s
}
