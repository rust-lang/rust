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

// Error messages for EXXXX errors.  Each message should start and end with a
// new line, and be wrapped to 80 characters.  In vim you can `:set tw=80` and
// use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {

E0154: r##"
Imports (`use` statements) are not allowed after non-item statements, such as
variable declarations and expression statements.

Here is an example that demonstrates the error:

```ignore
fn f() {
    // Variable declaration before import
    let x = 0;
    use std::io::Read;
    // ...
}
```

The solution is to declare the imports at the top of the block, function, or
file.

Here is the previous example again, with the correct order:

```
fn f() {
    use std::io::Read;
    let x = 0;
    // ...
}
```

See the Declaration Statements section of the reference for more information
about what constitutes an Item declaration and what does not:

https://doc.rust-lang.org/reference.html#statements
"##,

E0251: r##"
Two items of the same name cannot be imported without rebinding one of the
items under a new local name.

An example of this error:

```compile_fail
use foo::baz;
use bar::*; // error, do `use foo::baz as quux` instead on the previous line

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```
"##,

E0252: r##"
Two items of the same name cannot be imported without rebinding one of the
items under a new local name.

An example of this error:

```compile_fail
use foo::baz;
use bar::baz; // error, do `use bar::baz as quux` instead

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```
"##,

E0253: r##"
Attempt was made to import an unimportable value. This can happen when trying
to import a method from a trait. An example of this error:

```compile_fail
mod foo {
    pub trait MyTrait {
        fn do_something();
    }
}

use foo::MyTrait::do_something;
```

It's invalid to directly import methods belonging to a trait or concrete type.
"##,

E0255: r##"
You can't import a value whose name is the same as another value defined in the
module.

An example of this error:

```compile_fail
use bar::foo; // error, do `use bar::foo as baz` instead

fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {}
```
"##,

E0256: r##"
You can't import a type or module when the name of the item being imported is
the same as another type or submodule defined in the module.

An example of this error:

```compile_fail
use foo::Bar; // error

type Bar = u32;

mod foo {
    pub mod Bar { }
}

fn main() {}
```
"##,

E0259: r##"
The name chosen for an external crate conflicts with another external crate
that has been imported into the current module.

Erroneous code example:

```compile_fail
extern crate a;
extern crate crate_a as a;
```

The solution is to choose a different name that doesn't conflict with any
external crate imported into the current module.

Correct example:

```ignore
extern crate a;
extern crate crate_a as other_name;
```
"##,

E0260: r##"
The name for an item declaration conflicts with an external crate's name.

For instance:

```ignore
extern crate abc;

struct abc;
```

There are two possible solutions:

Solution #1: Rename the item.

```ignore
extern crate abc;

struct xyz;
```

Solution #2: Import the crate with a different name.

```ignore
extern crate abc as xyz;

struct abc;
```

See the Declaration Statements section of the reference for more information
about what constitutes an Item declaration and what does not:

https://doc.rust-lang.org/reference.html#statements
"##,

E0364: r##"
Private items cannot be publicly re-exported.  This error indicates that you
attempted to `pub use` a type or value that was not itself public.

Here is an example that demonstrates the error:

```compile_fail
mod foo {
    const X: u32 = 1;
}

pub use foo::X;
```

The solution to this problem is to ensure that the items that you are
re-exporting are themselves marked with `pub`:

```ignore
mod foo {
    pub const X: u32 = 1;
}

pub use foo::X;
```

See the 'Use Declarations' section of the reference for more information on
this topic:

https://doc.rust-lang.org/reference.html#use-declarations
"##,

E0365: r##"
Private modules cannot be publicly re-exported. This error indicates that you
attempted to `pub use` a module that was not itself public.

Here is an example that demonstrates the error:

```compile_fail
mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;
```

The solution to this problem is to ensure that the module that you are
re-exporting is itself marked with `pub`:

```ignore
pub mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;
```

See the 'Use Declarations' section of the reference for more information
on this topic:

https://doc.rust-lang.org/reference.html#use-declarations
"##,

E0401: r##"
Inner items do not inherit type parameters from the functions they are embedded
in. For example, this will not compile:

```compile_fail
fn foo<T>(x: T) {
    fn bar(y: T) { // T is defined in the "outer" function
        // ..
    }
    bar(x);
}
```

Nor will this:

```compile_fail
fn foo<T>(x: T) {
    type MaybeT = Option<T>;
    // ...
}
```

Or this:

```compile_fail
fn foo<T>(x: T) {
    struct Foo {
        x: T,
    }
    // ...
}
```

Items inside functions are basically just like top-level items, except
that they can only be used from the function they are in.

There are a couple of solutions for this.

If the item is a function, you may use a closure:

```
fn foo<T>(x: T) {
    let bar = |y: T| { // explicit type annotation may not be necessary
        // ..
    };
    bar(x);
}
```

For a generic item, you can copy over the parameters:

```
fn foo<T>(x: T) {
    fn bar<T>(y: T) {
        // ..
    }
    bar(x);
}
```

```
fn foo<T>(x: T) {
    type MaybeT<T> = Option<T>;
}
```

Be sure to copy over any bounds as well:

```
fn foo<T: Copy>(x: T) {
    fn bar<T: Copy>(y: T) {
        // ..
    }
    bar(x);
}
```

```
fn foo<T: Copy>(x: T) {
    struct Foo<T: Copy> {
        x: T,
    }
}
```

This may require additional type hints in the function body.

In case the item is a function inside an `impl`, defining a private helper
function might be easier:

```ignore
impl<T> Foo<T> {
    pub fn foo(&self, x: T) {
        self.bar(x);
    }

    fn bar(&self, y: T) {
        // ..
    }
}
```

For default impls in traits, the private helper solution won't work, however
closures or copying the parameters should still work.
"##,

E0403: r##"
Some type parameters have the same name. Example of erroneous code:

```compile_fail
fn foo<T, T>(s: T, u: T) {} // error: the name `T` is already used for a type
                            //        parameter in this type parameter list
```

Please verify that none of the type parameterss are misspelled, and rename any
clashing parameters. Example:

```
fn foo<T, Y>(s: T, u: Y) {} // ok!
```
"##,

E0404: r##"
You tried to implement something which was not a trait on an object. Example of
erroneous code:

```compile_fail
struct Foo;
struct Bar;

impl Foo for Bar {} // error: `Foo` is not a trait
```

Please verify that you didn't misspell the trait's name or otherwise use the
wrong identifier. Example:

```
trait Foo {
    // some functions
}
struct Bar;

impl Foo for Bar { // ok!
    // functions implementation
}
```
"##,

E0405: r##"
The code refers to a trait that is not in scope. Example of erroneous code:

```compile_fail
struct Foo;

impl SomeTrait for Foo {} // error: trait `SomeTrait` is not in scope
```

Please verify that the name of the trait wasn't misspelled and ensure that it
was imported. Example:

```ignore
// solution 1:
use some_file::SomeTrait;

// solution 2:
trait SomeTrait {
    // some functions
}

struct Foo;

impl SomeTrait for Foo { // ok!
    // implements functions
}
```
"##,

E0407: r##"
A definition of a method not in the implemented trait was given in a trait
implementation. Example of erroneous code:

```compile_fail
trait Foo {
    fn a();
}

struct Bar;

impl Foo for Bar {
    fn a() {}
    fn b() {} // error: method `b` is not a member of trait `Foo`
}
```

Please verify you didn't misspell the method name and you used the correct
trait. First example:

```
trait Foo {
    fn a();
    fn b();
}

struct Bar;

impl Foo for Bar {
    fn a() {}
    fn b() {} // ok!
}
```

Second example:

```
trait Foo {
    fn a();
}

struct Bar;

impl Foo for Bar {
    fn a() {}
}

impl Bar {
    fn b() {}
}
```
"##,

E0408: r##"
An "or" pattern was used where the variable bindings are not consistently bound
across patterns.

Example of erroneous code:

```compile_fail
match x {
    Some(y) | None => { /* use y */ } // error: variable `y` from pattern #1 is
                                      //        not bound in pattern #2
    _ => ()
}
```

Here, `y` is bound to the contents of the `Some` and can be used within the
block corresponding to the match arm. However, in case `x` is `None`, we have
not specified what `y` is, and the block will use a nonexistent variable.

To fix this error, either split into multiple match arms:

```
let x = Some(1);
match x {
    Some(y) => { /* use y */ }
    None => { /* ... */ }
}
```

or, bind the variable to a field of the same type in all sub-patterns of the
or pattern:

```
let x = (0, 2);
match x {
    (0, y) | (y, 0) => { /* use y */}
    _ => {}
}
```

In this example, if `x` matches the pattern `(0, _)`, the second field is set
to `y`. If it matches `(_, 0)`, the first field is set to `y`; so in all
cases `y` is set to some value.
"##,

E0409: r##"
An "or" pattern was used where the variable bindings are not consistently bound
across patterns.

Example of erroneous code:

```compile_fail
let x = (0, 2);
match x {
    (0, ref y) | (y, 0) => { /* use y */} // error: variable `y` is bound with
                                          //        different mode in pattern #2
                                          //        than in pattern #1
    _ => ()
}
```

Here, `y` is bound by-value in one case and by-reference in the other.

To fix this error, just use the same mode in both cases.
Generally using `ref` or `ref mut` where not already used will fix this:

```ignore
let x = (0, 2);
match x {
    (0, ref y) | (ref y, 0) => { /* use y */}
    _ => ()
}
```

Alternatively, split the pattern:

```
let x = (0, 2);
match x {
    (y, 0) => { /* use y */ }
    (0, ref y) => { /* use y */}
    _ => ()
}
```
"##,

E0411: r##"
The `Self` keyword was used outside an impl or a trait. Erroneous code example:

```compile_fail
<Self>::foo; // error: use of `Self` outside of an impl or trait
```

The `Self` keyword represents the current type, which explains why it can only
be used inside an impl or a trait. It gives access to the associated items of a
type:

```
trait Foo {
    type Bar;
}

trait Baz : Foo {
    fn bar() -> Self::Bar; // like this
}
```

However, be careful when two types have a common associated type:

```compile_fail
trait Foo {
    type Bar;
}

trait Foo2 {
    type Bar;
}

trait Baz : Foo + Foo2 {
    fn bar() -> Self::Bar;
    // error: ambiguous associated type `Bar` in bounds of `Self`
}
```

This problem can be solved by specifying from which trait we want to use the
`Bar` type:

```
trait Foo {
    type Bar;
}

trait Foo2 {
    type Bar;
}

trait Baz : Foo + Foo2 {
    fn bar() -> <Self as Foo>::Bar; // ok!
}
```
"##,

E0412: r##"
The type name used is not in scope. Example of erroneous codes:

```compile_fail
impl Something {} // error: type name `Something` is not in scope

// or:

trait Foo {
    fn bar(N); // error: type name `N` is not in scope
}

// or:

fn foo(x: T) {} // type name `T` is not in scope
```

To fix this error, please verify you didn't misspell the type name, you did
declare it or imported it into the scope. Examples:

```
struct Something;

impl Something {} // ok!

// or:

trait Foo {
    type N;

    fn bar(Self::N); // ok!
}

// or:

fn foo<T>(x: T) {} // ok!
```
"##,

E0413: r##"
A declaration shadows an enum variant or unit-like struct in scope. Example of
erroneous code:

```compile_fail
struct Foo;

let Foo = 12i32; // error: declaration of `Foo` shadows an enum variant or
                 //        unit-like struct in scope
```

To fix this error, rename the variable such that it doesn't shadow any enum
variable or structure in scope. Example:

```
struct Foo;

let foo = 12i32; // ok!
```

Or:

```
struct FooStruct;

let Foo = 12i32; // ok!
```

The goal here is to avoid a conflict of names.
"##,

E0414: r##"
A variable binding in an irrefutable pattern is shadowing the name of a
constant. Example of erroneous code:

```compile_fail
const FOO: u8 = 7;

let FOO = 5; // error: variable bindings cannot shadow constants

// or

fn bar(FOO: u8) { // error: variable bindings cannot shadow constants

}

// or

for FOO in bar {

}
```

Introducing a new variable in Rust is done through a pattern. Thus you can have
`let` bindings like `let (a, b) = ...`. However, patterns also allow constants
in them, e.g. if you want to match over a constant:

```ignore
const FOO: u8 = 1;

match (x,y) {
 (3, 4) => { .. }, // it is (3,4)
 (FOO, 1) => { .. }, // it is (1,1)
 (foo, 1) => { .. }, // it is (anything, 1)
                     // call the value in the first slot "foo"
 _ => { .. } // it is anything
}
```

Here, the second arm matches the value of `x` against the constant `FOO`,
whereas the third arm will accept any value of `x` and call it `foo`.

This works for `match`, however in cases where an irrefutable pattern is
required, constants can't be used. An irrefutable pattern is one which always
matches, whose purpose is only to bind variable names to values. These are
required by let, for, and function argument patterns.

Refutable patterns in such a situation do not make sense, for example:

```ignore
let Some(x) = foo; // what if foo is None, instead?

let (1, x) = foo; // what if foo.0 is not 1?

let (SOME_CONST, x) = foo; // what if foo.0 is not SOME_CONST?

let SOME_CONST = foo; // what if foo is not SOME_CONST?
```

Thus, an irrefutable variable binding can't contain a constant.

To fix this error, just give the marked variable a different name.
"##,

E0415: r##"
More than one function parameter have the same name. Example of erroneous code:

```compile_fail
fn foo(f: i32, f: i32) {} // error: identifier `f` is bound more than
                          //        once in this parameter list
```

Please verify you didn't misspell parameters' name. Example:

```
fn foo(f: i32, g: i32) {} // ok!
```
"##,

E0416: r##"
An identifier is bound more than once in a pattern. Example of erroneous code:

```compile_fail
match (1, 2) {
    (x, x) => {} // error: identifier `x` is bound more than once in the
                 //        same pattern
}
```

Please verify you didn't misspell identifiers' name. Example:

```
match (1, 2) {
    (x, y) => {} // ok!
}
```

Or maybe did you mean to unify? Consider using a guard:

```ignore
match (A, B, C) {
    (x, x2, see) if x == x2 => { /* A and B are equal, do one thing */ }
    (y, z, see) => { /* A and B unequal; do another thing */ }
}
```
"##,

E0417: r##"
A static variable was referenced in a pattern. Example of erroneous code:

```compile_fail
static FOO : i32 = 0;

match 0 {
    FOO => {} // error: static variables cannot be referenced in a
              //        pattern, use a `const` instead
    _ => {}
}
```

The compiler needs to know the value of the pattern at compile time;
compile-time patterns can defined via const or enum items. Please verify
that the identifier is spelled correctly, and if so, use a const instead
of static to define it. Example:

```
const FOO : i32 = 0;

match 0 {
    FOO => {} // ok!
    _ => {}
}
```
"##,

E0419: r##"
An unknown enum variant, struct or const was used. Example of erroneous code:

```compile_fail
match 0 {
    Something::Foo => {} // error: unresolved enum variant, struct
                         //        or const `Foo`
}
```

Please verify you didn't misspell it and the enum variant, struct or const has
been declared and imported into scope. Example:

```
enum Something {
    Foo,
    NotFoo,
}

match Something::NotFoo {
    Something::Foo => {} // ok!
    _ => {}
}
```
"##,

E0422: r##"
You are trying to use an identifier that is either undefined or not a struct.
For instance:

``` compile_fail
fn main () {
    let x = Foo { x: 1, y: 2 };
}
```

In this case, `Foo` is undefined, so it inherently isn't anything, and
definitely not a struct.

```compile_fail
fn main () {
    let foo = 1;
    let x = foo { x: 1, y: 2 };
}
```

In this case, `foo` is defined, but is not a struct, so Rust can't use it as
one.
"##,

E0423: r##"
A `struct` variant name was used like a function name. Example of erroneous
code:

```compile_fail
struct Foo { a: bool};

let f = Foo();
// error: `Foo` is a struct variant name, but this expression uses
//        it like a function name
```

Please verify you didn't misspell the name of what you actually wanted to use
here. Example:

```
fn Foo() -> u32 { 0 }

let f = Foo(); // ok!
```
"##,

E0424: r##"
The `self` keyword was used in a static method. Example of erroneous code:

```compile_fail
struct Foo;

impl Foo {
    fn bar(self) {}

    fn foo() {
        self.bar(); // error: `self` is not available in a static method.
    }
}
```

Please check if the method's argument list should have contained `self`,
`&self`, or `&mut self` (in case you didn't want to create a static
method), and add it if so. Example:

```
struct Foo;

impl Foo {
    fn bar(self) {}

    fn foo(self) {
        self.bar(); // ok!
    }
}
```
"##,

E0425: r##"
An unresolved name was used. Example of erroneous codes:

```compile_fail
something_that_doesnt_exist::foo;
// error: unresolved name `something_that_doesnt_exist::foo`

// or:

trait Foo {
    fn bar() {
        Self; // error: unresolved name `Self`
    }
}

// or:

let x = unknown_variable;  // error: unresolved name `unknown_variable`
```

Please verify that the name wasn't misspelled and ensure that the
identifier being referred to is valid for the given situation. Example:

```
enum something_that_does_exist {
    Foo,
}
```

Or:

```
mod something_that_does_exist {
    pub static foo : i32 = 0i32;
}

something_that_does_exist::foo; // ok!
```

Or:

```
let unknown_variable = 12u32;
let x = unknown_variable; // ok!
```
"##,

E0426: r##"
An undeclared label was used. Example of erroneous code:

```compile_fail
loop {
    break 'a; // error: use of undeclared label `'a`
}
```

Please verify you spelt or declare the label correctly. Example:

```
'a: loop {
    break 'a; // ok!
}
```
"##,

E0428: r##"
A type or module has been defined more than once. Example of erroneous
code:

```compile_fail
struct Bar;
struct Bar; // error: duplicate definition of value `Bar`
```

Please verify you didn't misspell the type/module's name or remove/rename the
duplicated one. Example:

```
struct Bar;
struct Bar2; // ok!
```
"##,

E0430: r##"
The `self` import appears more than once in the list. Erroneous code example:

```compile_fail
use something::{self, self}; // error: `self` import can only appear once in
                             //        the list
```

Please verify you didn't misspell the import name or remove the duplicated
`self` import. Example:

```ignore
use something::self; // ok!
```
"##,

E0431: r##"
An invalid `self` import was made. Erroneous code example:

```compile_fail
use {self}; // error: `self` import can only appear in an import list with a
            //        non-empty prefix
```

You cannot import the current module into itself, please remove this import
or verify you didn't misspell it.
"##,

E0432: r##"
An import was unresolved. Erroneous code example:

```compile_fail
use something::Foo; // error: unresolved import `something::Foo`.
```

Paths in `use` statements are relative to the crate root. To import items
relative to the current and parent modules, use the `self::` and `super::`
prefixes, respectively. Also verify that you didn't misspell the import
name and that the import exists in the module from where you tried to
import it. Example:

```ignore
use self::something::Foo; // ok!

mod something {
    pub struct Foo;
}
```

Or, if you tried to use a module from an external crate, you may have missed
the `extern crate` declaration (which is usually placed in the crate root):

```ignore
extern crate homura; // Required to use the `homura` crate

use homura::Madoka;
```
"##,

E0433: r##"
Invalid import. Example of erroneous code:

```compile_fail
use something_which_doesnt_exist;
// error: unresolved import `something_which_doesnt_exist`
```

Please verify you didn't misspell the import's name.
"##,

E0434: r##"
This error indicates that a variable usage inside an inner function is invalid
because the variable comes from a dynamic environment. Inner functions do not
have access to their containing environment.

Example of erroneous code:

```compile_fail
fn foo() {
    let y = 5;
    fn bar() -> u32 {
        y // error: can't capture dynamic environment in a fn item; use the
          //        || { ... } closure form instead.
    }
}
```

Functions do not capture local variables. To fix this error, you can replace the
function with a closure:

```
fn foo() {
    let y = 5;
    let bar = || {
        y
    };
}
```

or replace the captured variable with a constant or a static item:

```
fn foo() {
    static mut X: u32 = 4;
    const Y: u32 = 5;
    fn bar() -> u32 {
        unsafe {
            X = 3;
        }
        Y
    }
}
```
"##,

E0435: r##"
A non-constant value was used to initialise a constant. Example of erroneous
code:

```compile_fail
let foo = 42u32;
const FOO : u32 = foo; // error: attempt to use a non-constant value in a
                       //        constant
```

To fix this error, please replace the value with a constant. Example:

```
const FOO : u32 = 42u32; // ok!
```

Or:

```
const OTHER_FOO : u32 = 42u32;
const FOO : u32 = OTHER_FOO; // ok!
```
"##,

E0437: r##"
Trait implementations can only implement associated types that are members of
the trait in question. This error indicates that you attempted to implement
an associated type whose name does not match the name of any associated type
in the trait.

Here is an example that demonstrates the error:

```compile_fail
trait Foo {}

impl Foo for i32 {
    type Bar = bool;
}
```

The solution to this problem is to remove the extraneous associated type:

```
trait Foo {}

impl Foo for i32 {}
```
"##,

E0438: r##"
Trait implementations can only implement associated constants that are
members of the trait in question. This error indicates that you
attempted to implement an associated constant whose name does not
match the name of any associated constant in the trait.

Here is an example that demonstrates the error:

```compile_fail
#![feature(associated_consts)]

trait Foo {}

impl Foo for i32 {
    const BAR: bool = true;
}
```

The solution to this problem is to remove the extraneous associated constant:

```
trait Foo {}

impl Foo for i32 {}
```
"##

}

register_diagnostics! {
//  E0153, unused error code
//  E0157, unused error code
    E0254, // import conflicts with imported crate in this module
//  E0257,
//  E0258,
    E0402, // cannot use an outer type parameter in this context
    E0406, // undeclared associated type
//  E0410, merged into 408
    E0418, // is not an enum variant, struct or const
    E0420, // is not an associated const
    E0421, // unresolved associated const
    E0427, // cannot use `ref` binding mode with ...
    E0429, // `self` imports are only allowed within a { } list
}
