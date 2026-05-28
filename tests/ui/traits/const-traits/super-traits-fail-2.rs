//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]
//@ revisions: yy yn ny nn

#[cfg(any(yy, yn))] const trait Foo { fn a(&self); }
#[cfg(any(ny, nn))] trait Foo { fn a(&self); }

#[cfg(any(yy, ny))] const trait Bar: [const] Foo {}
//[ny]~^ ERROR: `[const]` can only be applied to `const` traits
//[ny]~| ERROR: `[const]` can only be applied to `const` traits
//[ny]~| ERROR: `[const]` can only be applied to `const` traits
//[ny]~| ERROR: `[const]` can only be applied to `const` traits
//[ny]~| ERROR: `[const]` can only be applied to `const` traits
#[cfg(any(yn, nn))] trait Bar: [const] Foo {}
//[yn,nn]~^ ERROR: `[const]` is not allowed here
//[nn]~^^ ERROR: `[const]` can only be applied to `const` traits
//[nn]~| ERROR: `[const]` can only be applied to `const` traits
//[nn]~| ERROR: `[const]` can only be applied to `const` traits

const fn foo<T: Bar>(x: &T) {
    x.a();
    //[yy,yn]~^ ERROR the trait bound `T: [const] Foo`
    //[nn,ny]~^^ ERROR cannot call non-const method `<T as Foo>::a` in constant functions
}

fn main() {}
