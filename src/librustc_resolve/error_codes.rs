#![allow(non_snake_case)]

use syntax::{register_diagnostic, register_diagnostics, register_long_diagnostics};

// Error messages for EXXXX errors.  Each message should start and end with a
// new line, and be wrapped to 80 characters.  In vim you can `:set tw=80` and
// use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {

E0128: r##"
Type parameter defaults can only use parameters that occur before them.
Erroneous code example:

```compile_fail,E0128
struct Foo<T=U, U=()> {
    field1: T,
    filed2: U,
}
// error: type parameters with a default cannot use forward declared
// identifiers
```

Since type parameters are evaluated in-order, you may be able to fix this issue
by doing:

```
struct Foo<U=(), T=U> {
    field1: T,
    filed2: U,
}
```

Please also verify that this wasn't because of a name-clash and rename the type
parameter if so.
"##,

E0154: r##"
#### Note: this error code is no longer emitted by the compiler.

Imports (`use` statements) are not allowed after non-item statements, such as
variable declarations and expression statements.

Here is an example that demonstrates the error:

```
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
#### Note: this error code is no longer emitted by the compiler.

Two items of the same name cannot be imported without rebinding one of the
items under a new local name.

An example of this error:

```
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

Erroneous code example:

```compile_fail,E0252
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

You can use aliases in order to fix this error. Example:

```
use foo::baz as foo_baz;
use bar::baz; // ok!

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```

Or you can reference the item with its parent:

```
use bar::baz;

fn main() {
    let x = foo::baz; // ok!
}

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
to import a method from a trait.

Erroneous code example:

```compile_fail,E0253
mod foo {
    pub trait MyTrait {
        fn do_something();
    }
}

use foo::MyTrait::do_something;
// error: `do_something` is not directly importable

fn main() {}
```

It's invalid to directly import methods belonging to a trait or concrete type.
"##,

E0254: r##"
Attempt was made to import an item whereas an extern crate with this name has
already been imported.

Erroneous code example:

```compile_fail,E0254
extern crate core;

mod foo {
    pub trait core {
        fn do_something();
    }
}

use foo::core;  // error: an extern crate named `core` has already
                //        been imported in this module

fn main() {}
```

To fix this issue, you have to rename at least one of the two imports.
Example:

```
extern crate core as libcore; // ok!

mod foo {
    pub trait core {
        fn do_something();
    }
}

use foo::core;

fn main() {}
```
"##,

E0255: r##"
You can't import a value whose name is the same as another value defined in the
module.

Erroneous code example:

```compile_fail,E0255
use bar::foo; // error: an item named `foo` is already in scope

fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {}
```

You can use aliases in order to fix this error. Example:

```
use bar::foo as bar_foo; // ok!

fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {}
```

Or you can reference the item with its parent:

```
fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {
    bar::foo(); // we get the item by referring to its parent
}
```
"##,

E0256: r##"
#### Note: this error code is no longer emitted by the compiler.

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

```compile_fail,E0259
extern crate core;
extern crate std as core;

fn main() {}
```

The solution is to choose a different name that doesn't conflict with any
external crate imported into the current module.

Correct example:

```
extern crate core;
extern crate std as other_name;

fn main() {}
```
"##,

E0260: r##"
The name for an item declaration conflicts with an external crate's name.

Erroneous code example:

```compile_fail,E0260
extern crate core;

struct core;

fn main() {}
```

There are two possible solutions:

Solution #1: Rename the item.

```
extern crate core;

struct xyz;
```

Solution #2: Import the crate with a different name.

```
extern crate core as xyz;

struct abc;
```

See the Declaration Statements section of the reference for more information
about what constitutes an Item declaration and what does not:

https://doc.rust-lang.org/reference.html#statements
"##,

E0364: r##"
Private items cannot be publicly re-exported. This error indicates that you
attempted to `pub use` a type or value that was not itself public.

Erroneous code example:

```compile_fail
mod foo {
    const X: u32 = 1;
}

pub use foo::X;

fn main() {}
```

The solution to this problem is to ensure that the items that you are
re-exporting are themselves marked with `pub`:

```
mod foo {
    pub const X: u32 = 1;
}

pub use foo::X;

fn main() {}
```

See the 'Use Declarations' section of the reference for more information on
this topic:

https://doc.rust-lang.org/reference.html#use-declarations
"##,

E0365: r##"
Private modules cannot be publicly re-exported. This error indicates that you
attempted to `pub use` a module that was not itself public.

Erroneous code example:

```compile_fail,E0365
mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;

fn main() {}
```

The solution to this problem is to ensure that the module that you are
re-exporting is itself marked with `pub`:

```
pub mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;

fn main() {}
```

See the 'Use Declarations' section of the reference for more information
on this topic:

https://doc.rust-lang.org/reference.html#use-declarations
"##,

E0401: r##"
Inner items do not inherit type or const parameters from the functions
they are embedded in.

Erroneous code example:

```compile_fail,E0401
fn foo<T>(x: T) {
    fn bar(y: T) { // T is defined in the "outer" function
        // ..
    }
    bar(x);
}
```

Nor will this:

```compile_fail,E0401
fn foo<T>(x: T) {
    type MaybeT = Option<T>;
    // ...
}
```

Or this:

```compile_fail,E0401
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

```
# struct Foo<T>(T);
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
Some type parameters have the same name.

Erroneous code example:

```compile_fail,E0403
fn foo<T, T>(s: T, u: T) {} // error: the name `T` is already used for a type
                            //        parameter in this type parameter list
```

Please verify that none of the type parameters are misspelled, and rename any
clashing parameters. Example:

```
fn foo<T, Y>(s: T, u: Y) {} // ok!
```
"##,

E0404: r##"
You tried to use something which is not a trait in a trait position, such as
a bound or `impl`.

Erroneous code example:

```compile_fail,E0404
struct Foo;
struct Bar;

impl Foo for Bar {} // error: `Foo` is not a trait
```

Another erroneous code example:

```compile_fail,E0404
struct Foo;

fn bar<T: Foo>(t: T) {} // error: `Foo` is not a trait
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

or

```
trait Foo {
    // some functions
}

fn bar<T: Foo>(t: T) {} // ok!
```

"##,

E0405: r##"
The code refers to a trait that is not in scope.

Erroneous code example:

```compile_fail,E0405
struct Foo;

impl SomeTrait for Foo {} // error: trait `SomeTrait` is not in scope
```

Please verify that the name of the trait wasn't misspelled and ensure that it
was imported. Example:

```
# #[cfg(for_demonstration_only)]
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
implementation.

Erroneous code example:

```compile_fail,E0407
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

Erroneous code example:

```compile_fail,E0408
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

Erroneous code example:

```compile_fail,E0409
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

```
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
The `Self` keyword was used outside an impl, trait, or type definition.

Erroneous code example:

```compile_fail,E0411
<Self>::foo; // error: use of `Self` outside of an impl, trait, or type
             // definition
```

The `Self` keyword represents the current type, which explains why it can only
be used inside an impl, trait, or type definition. It gives access to the
associated items of a type:

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
The type name used is not in scope.

Erroneous code examples:

```compile_fail,E0412
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

    fn bar(_: Self::N); // ok!
}

// or:

fn foo<T>(x: T) {} // ok!
```

Another case that causes this error is when a type is imported into a parent
module. To fix this, you can follow the suggestion and use File directly or
`use super::File;` which will import the types from the parent namespace. An
example that causes this error is below:

```compile_fail,E0412
use std::fs::File;

mod foo {
    fn some_function(f: File) {}
}
```

```
use std::fs::File;

mod foo {
    // either
    use super::File;
    // or
    // use std::fs::File;
    fn foo(f: File) {}
}
# fn main() {} // don't insert it for us; that'll break imports
```
"##,

E0415: r##"
More than one function parameter have the same name.

Erroneous code example:

```compile_fail,E0415
fn foo(f: i32, f: i32) {} // error: identifier `f` is bound more than
                          //        once in this parameter list
```

Please verify you didn't misspell parameters' name. Example:

```
fn foo(f: i32, g: i32) {} // ok!
```
"##,

E0416: r##"
An identifier is bound more than once in a pattern.

Erroneous code example:

```compile_fail,E0416
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

```
# let (A, B, C) = (1, 2, 3);
match (A, B, C) {
    (x, x2, see) if x == x2 => { /* A and B are equal, do one thing */ }
    (y, z, see) => { /* A and B unequal; do another thing */ }
}
```
"##,

E0422: r##"
You are trying to use an identifier that is either undefined or not a struct.
Erroneous code example:

```compile_fail,E0422
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
An identifier was used like a function name or a value was expected and the
identifier exists but it belongs to a different namespace.

For (an erroneous) example, here a `struct` variant name were used as a
function:

```compile_fail,E0423
struct Foo { a: bool };

let f = Foo();
// error: expected function, found `Foo`
// `Foo` is a struct name, but this expression uses it like a function name
```

Please verify you didn't misspell the name of what you actually wanted to use
here. Example:

```
fn Foo() -> u32 { 0 }

let f = Foo(); // ok!
```

It is common to forget the trailing `!` on macro invocations, which would also
yield this error:

```compile_fail,E0423
println("");
// error: expected function, found macro `println`
// did you mean `println!(...)`? (notice the trailing `!`)
```

Another case where this error is emitted is when a value is expected, but
something else is found:

```compile_fail,E0423
pub mod a {
    pub const I: i32 = 1;
}

fn h1() -> i32 {
    a.I
    //~^ ERROR expected value, found module `a`
    // did you mean `a::I`?
}
```
"##,

E0424: r##"
The `self` keyword was used in a static method.

Erroneous code example:

```compile_fail,E0424
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
An unresolved name was used.

Erroneous code examples:

```compile_fail,E0425
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

If the item is not defined in the current module, it must be imported using a
`use` statement, like so:

```
# mod foo { pub fn bar() {} }
# fn main() {
use foo::bar;
bar();
# }
```

If the item you are importing is not defined in some super-module of the
current module, then it must also be declared as public (e.g., `pub fn`).
"##,

E0426: r##"
An undeclared label was used.

Erroneous code example:

```compile_fail,E0426
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
A type or module has been defined more than once.

Erroneous code example:

```compile_fail,E0428
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

E0429: r##"
The `self` keyword cannot appear alone as the last segment in a `use`
declaration.

Erroneous code example:

```compile_fail,E0429
use std::fmt::self; // error: `self` imports are only allowed within a { } list
```

To use a namespace itself in addition to some of its members, `self` may appear
as part of a brace-enclosed list of imports:

```
use std::fmt::{self, Debug};
```

If you only want to import the namespace, do so directly:

```
use std::fmt;
```
"##,

E0430: r##"
The `self` import appears more than once in the list.

Erroneous code example:

```compile_fail,E0430
use something::{self, self}; // error: `self` import can only appear once in
                             //        the list
```

Please verify you didn't misspell the import name or remove the duplicated
`self` import. Example:

```
# mod something {}
# fn main() {
use something::{self}; // ok!
# }
```
"##,

E0431: r##"
An invalid `self` import was made.

Erroneous code example:

```compile_fail,E0431
use {self}; // error: `self` import can only appear in an import list with a
            //        non-empty prefix
```

You cannot import the current module into itself, please remove this import
or verify you didn't misspell it.
"##,

E0432: r##"
An import was unresolved.

Erroneous code example:

```compile_fail,E0432
use something::Foo; // error: unresolved import `something::Foo`.
```

Paths in `use` statements are relative to the crate root. To import items
relative to the current and parent modules, use the `self::` and `super::`
prefixes, respectively. Also verify that you didn't misspell the import
name and that the import exists in the module from where you tried to
import it. Example:

```
use self::something::Foo; // ok!

mod something {
    pub struct Foo;
}
# fn main() {}
```

Or, if you tried to use a module from an external crate, you may have missed
the `extern crate` declaration (which is usually placed in the crate root):

```
extern crate core; // Required to use the `core` crate

use core::any;
# fn main() {}
```
"##,

E0433: r##"
An undeclared type or module was used.

Erroneous code example:

```compile_fail,E0433
let map = HashMap::new();
// error: failed to resolve: use of undeclared type or module `HashMap`
```

Please verify you didn't misspell the type/module's name or that you didn't
forget to import it:


```
use std::collections::HashMap; // HashMap has been imported.
let map: HashMap<u32, u32> = HashMap::new(); // So it can be used!
```
"##,

E0434: r##"
This error indicates that a variable usage inside an inner function is invalid
because the variable comes from a dynamic environment. Inner functions do not
have access to their containing environment.

Erroneous code example:

```compile_fail,E0434
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
A non-constant value was used in a constant expression.

Erroneous code example:

```compile_fail,E0435
let foo = 42;
let a: [u8; foo]; // error: attempt to use a non-constant value in a constant
```

To fix this error, please replace the value with a constant. Example:

```
let a: [u8; 42]; // ok!
```

Or:

```
const FOO: usize = 42;
let a: [u8; FOO]; // ok!
```
"##,

E0437: r##"
Trait implementations can only implement associated types that are members of
the trait in question. This error indicates that you attempted to implement
an associated type whose name does not match the name of any associated type
in the trait.

Erroneous code example:

```compile_fail,E0437
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

Erroneous code example:

```compile_fail,E0438
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
"##,

E0466: r##"
Macro import declarations were malformed.

Erroneous code examples:

```compile_fail,E0466
#[macro_use(a_macro(another_macro))] // error: invalid import declaration
extern crate core as some_crate;

#[macro_use(i_want = "some_macros")] // error: invalid import declaration
extern crate core as another_crate;
```

This is a syntax error at the level of attribute declarations. The proper
syntax for macro imports is the following:

```ignore (cannot-doctest-multicrate-project)
// In some_crate:
#[macro_export]
macro_rules! get_tacos {
    ...
}

#[macro_export]
macro_rules! get_pimientos {
    ...
}

// In your crate:
#[macro_use(get_tacos, get_pimientos)] // It imports `get_tacos` and
extern crate some_crate;               // `get_pimientos` macros from some_crate
```

If you would like to import all exported macros, write `macro_use` with no
arguments.
"##,

E0468: r##"
A non-root module attempts to import macros from another crate.

Example of erroneous code:

```compile_fail,E0468
mod foo {
    #[macro_use(debug_assert)]  // error: must be at crate root to import
    extern crate core;          //        macros from another crate
    fn run_macro() { debug_assert!(true); }
}
```

Only `extern crate` imports at the crate root level are allowed to import
macros.

Either move the macro import to crate root or do without the foreign macros.
This will work:

```
#[macro_use(debug_assert)]
extern crate core;

mod foo {
    fn run_macro() { debug_assert!(true); }
}
# fn main() {}
```
"##,

E0469: r##"
A macro listed for import was not found.

Erroneous code example:

```compile_fail,E0469
#[macro_use(drink, be_merry)] // error: imported macro not found
extern crate alloc;

fn main() {
    // ...
}
```

Either the listed macro is not contained in the imported crate, or it is not
exported from the given crate.

This could be caused by a typo. Did you misspell the macro's name?

Double-check the names of the macros listed for import, and that the crate
in question exports them.

A working version would be:

```ignore (cannot-doctest-multicrate-project)
// In some_crate crate:
#[macro_export]
macro_rules! eat {
    ...
}

#[macro_export]
macro_rules! drink {
    ...
}

// In your crate:
#[macro_use(eat, drink)]
extern crate some_crate; //ok!
```
"##,

E0530: r##"
A binding shadowed something it shouldn't.

Erroneous code example:

```compile_fail,E0530
static TEST: i32 = 0;

let r: (i32, i32) = (0, 0);
match r {
    TEST => {} // error: match bindings cannot shadow statics
}
```

To fix this error, just change the binding's name in order to avoid shadowing
one of the following:

* struct name
* struct/enum variant
* static
* const
* associated const

Fixed example:

```
static TEST: i32 = 0;

let r: (i32, i32) = (0, 0);
match r {
    something => {} // ok!
}
```
"##,

E0532: r##"
Pattern arm did not match expected kind.

Erroneous code example:

```compile_fail,E0532
enum State {
    Succeeded,
    Failed(String),
}

fn print_on_failure(state: &State) {
    match *state {
        // error: expected unit struct/variant or constant, found tuple
        //        variant `State::Failed`
        State::Failed => println!("Failed"),
        _ => ()
    }
}
```

To fix this error, ensure the match arm kind is the same as the expression
matched.

Fixed example:

```
enum State {
    Succeeded,
    Failed(String),
}

fn print_on_failure(state: &State) {
    match *state {
        State::Failed(ref msg) => println!("Failed with {}", msg),
        _ => ()
    }
}
```
"##,

E0603: r##"
A private item was used outside its scope.

Erroneous code example:

```compile_fail,E0603
mod SomeModule {
    const PRIVATE: u32 = 0x_a_bad_1dea_u32; // This const is private, so we
                                            // can't use it outside of the
                                            // `SomeModule` module.
}

println!("const value: {}", SomeModule::PRIVATE); // error: constant `PRIVATE`
                                                  //        is private
```

In order to fix this error, you need to make the item public by using the `pub`
keyword. Example:

```
mod SomeModule {
    pub const PRIVATE: u32 = 0x_a_bad_1dea_u32; // We set it public by using the
                                                // `pub` keyword.
}

println!("const value: {}", SomeModule::PRIVATE); // ok!
```
"##,

E0659: r##"
An item usage is ambiguous.

Erroneous code example:

```compile_fail,E0659
pub mod moon {
    pub fn foo() {}
}

pub mod earth {
    pub fn foo() {}
}

mod collider {
    pub use moon::*;
    pub use earth::*;
}

fn main() {
    collider::foo(); // ERROR: `foo` is ambiguous
}
```

This error generally appears when two items with the same name are imported into
a module. Here, the `foo` functions are imported and reexported from the
`collider` module and therefore, when we're using `collider::foo()`, both
functions collide.

To solve this error, the best solution is generally to keep the path before the
item when using it. Example:

```
pub mod moon {
    pub fn foo() {}
}

pub mod earth {
    pub fn foo() {}
}

mod collider {
    pub use moon;
    pub use earth;
}

fn main() {
    collider::moon::foo(); // ok!
    collider::earth::foo(); // ok!
}
```
"##,

E0671: r##"
Const parameters cannot depend on type parameters.
The following is therefore invalid:
```compile_fail,E0671
#![feature(const_generics)]

fn const_id<T, const N: T>() -> T { // error: const parameter
                                    // depends on type parameter
    N
}
```
"##,

}

register_diagnostics! {
//  E0153, unused error code
//  E0157, unused error code
//  E0257,
//  E0258,
//  E0402, // cannot use an outer type parameter in this context
//  E0406, merged into 420
//  E0410, merged into 408
//  E0413, merged into 530
//  E0414, merged into 530
//  E0417, merged into 532
//  E0418, merged into 532
//  E0419, merged into 531
//  E0420, merged into 532
//  E0421, merged into 531
    E0531, // unresolved pattern path kind `name`
//  E0427, merged into 530
//  E0467, removed
//  E0470, removed
    E0573,
    E0574,
    E0575,
    E0576,
    E0577,
    E0578,
}
