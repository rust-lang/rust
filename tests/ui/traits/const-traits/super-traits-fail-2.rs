//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]
//@ revisions: yy yn ny nn

#[cfg_attr(any(yy, yn), const_trait)]
trait Foo {
    fn a(&self);
}

#[cfg_attr(any(yy, ny), const_trait)]
trait Bar: [const] Foo {}
//[ny,nn]~^ ERROR: `[const]` can only be applied to `const` traits
//[ny,nn]~| ERROR: `[const]` can only be applied to `const` traits
//[ny,nn]~| ERROR: `[const]` can only be applied to `const` traits
//[ny]~| ERROR: `[const]` can only be applied to `const` traits
//[ny]~| ERROR: `[const]` can only be applied to `const` traits
//[yn,nn]~^^^^^^ ERROR: `[const]` is not allowed here

const fn foo<T: Bar>(x: &T) {
    x.a();
    //[yy,yn]~^ ERROR the trait bound `T: [const] Foo`
    //[nn,ny]~^^ ERROR cannot call non-const method `<T as Foo>::a` in constant functions
}

fn main() {}
