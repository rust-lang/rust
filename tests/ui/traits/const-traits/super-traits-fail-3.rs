//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

//@ revisions: yy yn ny nn
//@[yy] check-pass

#[cfg_attr(any(yy, yn), const_trait)]
trait Foo {
    fn a(&self);
}

#[cfg_attr(any(yy, ny), const_trait)]
trait Bar: ~const Foo {}
//[ny,nn]~^ ERROR: `~const` can only be applied to `#[const_trait]`
//[ny,nn]~| ERROR: `~const` can only be applied to `#[const_trait]`
//[ny,nn]~| ERROR: `~const` can only be applied to `#[const_trait]`
//[ny]~| ERROR: `~const` can only be applied to `#[const_trait]`
//[ny]~| ERROR: `~const` can only be applied to `#[const_trait]`
//[yn,nn]~^^^^^^ ERROR: `~const` is not allowed here

const fn foo<T: ~const Bar>(x: &T) {
    //[yn,nn]~^ ERROR: `~const` can only be applied to `#[const_trait]`
    //[yn,nn]~| ERROR: `~const` can only be applied to `#[const_trait]`
    x.a();
    //[yn]~^ ERROR: the trait bound `T: ~const Foo` is not satisfied
    //[nn,ny]~^^ ERROR: cannot call non-const fn `<T as Foo>::a` in constant functions
}

fn main() {}
