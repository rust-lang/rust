// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_long_diagnostics! {

E0445: r##"
A private trait was used on a public type parameter bound. Erroneous code
examples:

```
trait Foo {
    fn dummy(&self) { }
}

pub trait Bar : Foo {} // error: private trait in public interface
pub struct Bar<T: Foo>(pub T); // same error
pub fn foo<T: Foo> (t: T) {} // same error
```

To solve this error, please ensure that the trait is also public. The trait
can be made inaccessible if necessary by placing it into a private inner module,
but it still has to be marked with `pub`.
Example:

```
pub trait Foo { // we set the Foo trait public
    fn dummy(&self) { }
}

pub trait Bar : Foo {} // ok!
pub struct Bar<T: Foo>(pub T); // ok!
pub fn foo<T: Foo> (t: T) {} // ok!
```
"##,

E0446: r##"
A private type was used in an public type signature. Erroneous code example:

```
mod Foo {
    struct Bar(u32);

    pub fn bar() -> Bar { // error: private type in public interface
        Bar(0)
    }
}
```

To solve this error, please ensure that the type is also public. The type
can be made inaccessible if necessary by placing it into a private inner module,
but it still has to be marked with `pub`.
Example:

```
mod Foo {
    pub struct Bar(u32); // we set the Bar type public

    pub fn bar() -> Bar { // ok!
        Bar(0)
    }
}
```
"##,

E0447: r##"
The `pub` keyword was used inside a function. Erroneous code example:

```
fn foo() {
    pub struct Bar; // error: visibility has no effect inside functions
}
```

Since we cannot access items defined inside a function, the visibility of its
items does not impact outer code. So using the `pub` keyword in this context
is invalid.
"##,

E0448: r##"
The `pub` keyword was used inside a public enum. Erroneous code example:

```
pub enum Foo {
    pub Bar, // error: unnecessary `pub` visibility
}
```

Since the enum is already public, adding `pub` on one its elements is
unnecessary. Example:

```
enum Foo {
    pub Bar, // ok!
}

// or:

pub enum Foo {
    Bar, // ok!
}
```
"##,

E0449: r##"
A visibility qualifier was used when it was unnecessary. Erroneous code
examples:

```
struct Bar;

trait Foo {
    fn foo();
}

pub impl Bar {} // error: unnecessary visibility qualifier

pub impl Foo for Bar { // error: unnecessary visibility qualifier
    pub fn foo() {} // error: unnecessary visibility qualifier
}
```

To fix this error, please remove the visibility qualifier when it is not
required. Example:

```
struct Bar;

trait Foo {
    fn foo();
}

// Directly implemented methods share the visibility of the type itself,
// so `pub` is unnecessary here
impl Bar {}

// Trait methods share the visibility of the trait, so `pub` is
// unnecessary in either case
pub impl Foo for Bar {
    pub fn foo() {}
}
```
"##,

E0450: r##"
A tuple constructor was invoked while some of its fields are private. Erroneous
code example:

```
mod Bar {
    pub struct Foo(isize);
}

let f = Bar::Foo(0); // error: cannot invoke tuple struct constructor with
                     //        private fields
```

To solve this issue, please ensure that all of the fields of the tuple struct
are public. Alternatively, provide a new() method to the tuple struct to
construct it from a given inner value. Example:

```
mod Bar {
    pub struct Foo(pub isize); // we set its field to public
}

let f = Bar::Foo(0); // ok!

// or:
mod bar {
    pub struct Foo(isize);

    impl Foo {
        pub fn new(x: isize) {
            Foo(x)
        }
    }
}

let f = bar::Foo::new(1);
```
"##,

E0451: r##"
A struct constructor with private fields was invoked. Erroneous code example:

```
mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize,
    }
}

let f = Bar::Foo{ a: 0, b: 0 }; // error: field `b` of struct `Bar::Foo`
                                //        is private
```

To fix this error, please ensure that all the fields of the struct, or
implement a function for easy instantiation. Examples:

```
mod Bar {
    pub struct Foo {
        pub a: isize,
        pub b: isize, // we set `b` field public
    }
}

let f = Bar::Foo{ a: 0, b: 0 }; // ok!

// or:
mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize, // still private
    }

    impl Foo {
        pub fn new() -> Foo { // we create a method to instantiate `Foo`
            Foo { a: 0, b: 0 }
        }
    }
}

let f = Bar::Foo::new(); // ok!
```
"##,

}
