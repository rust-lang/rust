// Test parsing for `default` where it doesn't belong.
// Specifically, we are interested in kinds of items or items in certain contexts.
// Also test item kinds in `extern` blocks and associated contexts which are not allowed there.

fn main() {}

#[cfg(FALSE)]
mod free_items {
    default extern crate foo; //~ ERROR item cannot be `default`
    default use foo; //~ ERROR item cannot be `default`
    default static foo: u8; //~ ERROR item cannot be `default`
    default const foo: u8; //~ ERROR item cannot be `default`
    default fn foo(); //~ ERROR item cannot be `default`
    default mod foo {} //~ ERROR item cannot be `default`
    default extern "C" {} //~ ERROR item cannot be `default`
    default type foo = u8; //~ ERROR item cannot be `default`
    default enum foo {} //~ ERROR item cannot be `default`
    default struct foo {} //~ ERROR item cannot be `default`
    default union foo {} //~ ERROR item cannot be `default`
    default trait foo {} //~ ERROR item cannot be `default`
    default trait foo = Ord; //~ ERROR item cannot be `default`
    default impl foo {}
    default!();
    default::foo::bar!();
    default default!(); //~ ERROR item cannot be `default`
    default default::foo::bar!(); //~ ERROR item cannot be `default`
    default macro foo {} //~ ERROR item cannot be `default`
    default macro_rules! foo {} //~ ERROR item cannot be `default`
}

#[cfg(FALSE)]
extern "C" {
    default extern crate foo; //~ ERROR item cannot be `default`
    //~^ ERROR extern crate not supported in `extern` block
    default use foo; //~ ERROR item cannot be `default`
    //~^ ERROR `use` import not supported in `extern` block
    default static foo: u8; //~ ERROR item cannot be `default`
    default const foo: u8; //~ ERROR item cannot be `default`
    //~^ ERROR extern items cannot be `const`
    default fn foo(); //~ ERROR item cannot be `default`
    default mod foo {} //~ ERROR item cannot be `default`
    //~^ ERROR module not supported in `extern` block
    default extern "C" {} //~ ERROR item cannot be `default`
    //~^ ERROR extern block not supported in `extern` block
    default type foo = u8; //~ ERROR item cannot be `default`
    default enum foo {} //~ ERROR item cannot be `default`
    //~^ ERROR enum not supported in `extern` block
    default struct foo {} //~ ERROR item cannot be `default`
    //~^ ERROR struct not supported in `extern` block
    default union foo {} //~ ERROR item cannot be `default`
    //~^ ERROR union not supported in `extern` block
    default trait foo {} //~ ERROR item cannot be `default`
    //~^ ERROR trait not supported in `extern` block
    default trait foo = Ord; //~ ERROR item cannot be `default`
    //~^ ERROR trait alias not supported in `extern` block
    default impl foo {}
    //~^ ERROR implementation not supported in `extern` block
    default!();
    default::foo::bar!();
    default default!(); //~ ERROR item cannot be `default`
    default default::foo::bar!(); //~ ERROR item cannot be `default`
    default macro foo {} //~ ERROR item cannot be `default`
    //~^ ERROR macro definition not supported in `extern` block
    default macro_rules! foo {} //~ ERROR item cannot be `default`
    //~^ ERROR macro definition not supported in `extern` block
}

#[cfg(FALSE)]
impl S {
    default extern crate foo;
    //~^ ERROR extern crate not supported in `trait` or `impl`
    default use foo;
    //~^ ERROR `use` import not supported in `trait` or `impl`
    default static foo: u8;
    //~^ ERROR associated `static` items are not allowed
    default const foo: u8;
    default fn foo();
    default mod foo {}
    //~^ ERROR module not supported in `trait` or `impl`
    default extern "C" {}
    //~^ ERROR extern block not supported in `trait` or `impl`
    default type foo = u8;
    default enum foo {}
    //~^ ERROR enum not supported in `trait` or `impl`
    default struct foo {}
    //~^ ERROR struct not supported in `trait` or `impl`
    default union foo {}
    //~^ ERROR union not supported in `trait` or `impl`
    default trait foo {}
    //~^ ERROR trait not supported in `trait` or `impl`
    default trait foo = Ord;
    //~^ ERROR trait alias not supported in `trait` or `impl`
    default impl foo {}
    //~^ ERROR implementation not supported in `trait` or `impl`
    default!();
    default::foo::bar!();
    default default!();
    default default::foo::bar!();
    default macro foo {}
    //~^ ERROR macro definition not supported in `trait` or `impl`
    default macro_rules! foo {}
    //~^ ERROR macro definition not supported in `trait` or `impl`
}

#[cfg(FALSE)]
trait T {
    default extern crate foo;
    //~^ ERROR extern crate not supported in `trait` or `impl`
    default use foo;
    //~^ ERROR `use` import not supported in `trait` or `impl`
    default static foo: u8;
    //~^ ERROR associated `static` items are not allowed
    default const foo: u8;
    default fn foo();
    default mod foo {}
    //~^ ERROR module not supported in `trait` or `impl`
    default extern "C" {}
    //~^ ERROR extern block not supported in `trait` or `impl`
    default type foo = u8;
    default enum foo {}
    //~^ ERROR enum not supported in `trait` or `impl`
    default struct foo {}
    //~^ ERROR struct not supported in `trait` or `impl`
    default union foo {}
    //~^ ERROR union not supported in `trait` or `impl`
    default trait foo {}
    //~^ ERROR trait not supported in `trait` or `impl`
    default trait foo = Ord;
    //~^ ERROR trait alias not supported in `trait` or `impl`
    default impl foo {}
    //~^ ERROR implementation not supported in `trait` or `impl`
    default!();
    default::foo::bar!();
    default default!();
    default default::foo::bar!();
    default macro foo {}
    //~^ ERROR macro definition not supported in `trait` or `impl`
    default macro_rules! foo {}
    //~^ ERROR macro definition not supported in `trait` or `impl`
}
