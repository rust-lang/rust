#### Note: this error code is no longer emitted by the compiler

You implemented a trait, overriding one or more of its associated types but did
not reimplement its default methods.

Example of erroneous code:

```
#![feature(associated_type_defaults)]

pub trait Foo {
    type Assoc = u8;
    fn bar(&self) {}
}

impl Foo for i32 {
    // error - the following trait items need to be reimplemented as
    //         `Assoc` was overridden: `bar`
    type Assoc = i32;
}
```

To fix this, add an implementation for each default method from the trait:

```
#![feature(associated_type_defaults)]

pub trait Foo {
    type Assoc = u8;
    fn bar(&self) {}
}

impl Foo for i32 {
    type Assoc = i32;
    fn bar(&self) {} // ok!
}
```
