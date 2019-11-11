// ignore-tidy-filelength

// Error messages for EXXXX errors.  Each message should start and end with a
// new line, and be wrapped to 80 characters.  In vim you can `:set tw=80` and
// use `gq` to wrap paragraphs. Use `:set tw=0` to disable.

crate::register_diagnostics! {

E0001: r##"
#### Note: this error code is no longer emitted by the compiler.

This error suggests that the expression arm corresponding to the noted pattern
will never be reached as for all possible values of the expression being
matched, one of the preceding patterns will match.

This means that perhaps some of the preceding patterns are too general, this
one is too specific or the ordering is incorrect.

For example, the following `match` block has too many arms:

```
match Some(0) {
    Some(bar) => {/* ... */}
    x => {/* ... */} // This handles the `None` case
    _ => {/* ... */} // All possible cases have already been handled
}
```

`match` blocks have their patterns matched in order, so, for example, putting
a wildcard arm above a more specific arm will make the latter arm irrelevant.

Ensure the ordering of the match arm is correct and remove any superfluous
arms.
"##,

E0002: r##"
#### Note: this error code is no longer emitted by the compiler.

This error indicates that an empty match expression is invalid because the type
it is matching on is non-empty (there exist values of this type). In safe code
it is impossible to create an instance of an empty type, so empty match
expressions are almost never desired. This error is typically fixed by adding
one or more cases to the match expression.

An example of an empty type is `enum Empty { }`. So, the following will work:

```
enum Empty {}

fn foo(x: Empty) {
    match x {
        // empty
    }
}
```

However, this won't:

```compile_fail
fn foo(x: Option<String>) {
    match x {
        // empty
    }
}
```
"##,

E0004: r##"
This error indicates that the compiler cannot guarantee a matching pattern for
one or more possible inputs to a match expression. Guaranteed matches are
required in order to assign values to match expressions, or alternatively,
determine the flow of execution.

Erroneous code example:

```compile_fail,E0004
enum Terminator {
    HastaLaVistaBaby,
    TalkToMyHand,
}

let x = Terminator::HastaLaVistaBaby;

match x { // error: non-exhaustive patterns: `HastaLaVistaBaby` not covered
    Terminator::TalkToMyHand => {}
}
```

If you encounter this error you must alter your patterns so that every possible
value of the input type is matched. For types with a small number of variants
(like enums) you should probably cover all cases explicitly. Alternatively, the
underscore `_` wildcard pattern can be added after all other patterns to match
"anything else". Example:

```
enum Terminator {
    HastaLaVistaBaby,
    TalkToMyHand,
}

let x = Terminator::HastaLaVistaBaby;

match x {
    Terminator::TalkToMyHand => {}
    Terminator::HastaLaVistaBaby => {}
}

// or:

match x {
    Terminator::TalkToMyHand => {}
    _ => {}
}
```
"##,

E0005: r##"
Patterns used to bind names must be irrefutable, that is, they must guarantee
that a name will be extracted in all cases.

Erroneous code example:

```compile_fail,E0005
let x = Some(1);
let Some(y) = x;
// error: refutable pattern in local binding: `None` not covered
```

If you encounter this error you probably need to use a `match` or `if let` to
deal with the possibility of failure. Example:

```
let x = Some(1);

match x {
    Some(y) => {
        // do something
    },
    None => {}
}

// or:

if let Some(y) = x {
    // do something
}
```
"##,

E0007: r##"
This error indicates that the bindings in a match arm would require a value to
be moved into more than one location, thus violating unique ownership. Code
like the following is invalid as it requires the entire `Option<String>` to be
moved into a variable called `op_string` while simultaneously requiring the
inner `String` to be moved into a variable called `s`.

Erroneous code example:

```compile_fail,E0007
let x = Some("s".to_string());

match x {
    op_string @ Some(s) => {}, // error: cannot bind by-move with sub-bindings
    None => {},
}
```

See also the error E0303.
"##,

E0009: r##"
In a pattern, all values that don't implement the `Copy` trait have to be bound
the same way. The goal here is to avoid binding simultaneously by-move and
by-ref.

This limitation may be removed in a future version of Rust.

Erroneous code example:

```compile_fail,E0009
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((y, ref z)) => {}, // error: cannot bind by-move and by-ref in the
                            //        same pattern
    None => panic!()
}
```

You have two solutions:

Solution #1: Bind the pattern's values the same way.

```
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((ref y, ref z)) => {},
    // or Some((y, z)) => {}
    None => panic!()
}
```

Solution #2: Implement the `Copy` trait for the `X` structure.

However, please keep in mind that the first solution should be preferred.

```
#[derive(Clone, Copy)]
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((y, ref z)) => {},
    None => panic!()
}
```
"##,

E0010: r##"
The value of statics and constants must be known at compile time, and they live
for the entire lifetime of a program. Creating a boxed value allocates memory on
the heap at runtime, and therefore cannot be done at compile time.

Erroneous code example:

```compile_fail,E0010
#![feature(box_syntax)]

const CON : Box<i32> = box 0;
```
"##,

E0013: r##"
Static and const variables can refer to other const variables. But a const
variable cannot refer to a static variable.

Erroneous code example:

```compile_fail,E0013
static X: i32 = 42;
const Y: i32 = X;
```

In this example, `Y` cannot refer to `X` here. To fix this, the value can be
extracted as a const and then used:

```
const A: i32 = 42;
static X: i32 = A;
const Y: i32 = A;
```
"##,

E0014: r##"
#### Note: this error code is no longer emitted by the compiler.

Constants can only be initialized by a constant value or, in a future
version of Rust, a call to a const function. This error indicates the use
of a path (like a::b, or x) denoting something other than one of these
allowed items.

Erroneous code example:

```
const FOO: i32 = { let x = 0; x }; // 'x' isn't a constant nor a function!
```

To avoid it, you have to replace the non-constant value:

```
const FOO: i32 = { const X : i32 = 0; X };
// or even:
const FOO2: i32 = { 0 }; // but brackets are useless here
```
"##,

E0015: r##"
The only functions that can be called in static or constant expressions are
`const` functions, and struct/enum constructors. `const` functions are only
available on a nightly compiler. Rust currently does not support more general
compile-time function execution.

```
const FOO: Option<u8> = Some(1); // enum constructor
struct Bar {x: u8}
const BAR: Bar = Bar {x: 1}; // struct constructor
```

See [RFC 911] for more details on the design of `const fn`s.

[RFC 911]: https://github.com/rust-lang/rfcs/blob/master/text/0911-const-fn.md
"##,

E0017: r##"
References in statics and constants may only refer to immutable values.

Erroneous code example:

```compile_fail,E0017
static X: i32 = 1;
const C: i32 = 2;

// these three are not allowed:
const CR: &mut i32 = &mut C;
static STATIC_REF: &'static mut i32 = &mut X;
static CONST_REF: &'static mut i32 = &mut C;
```

Statics are shared everywhere, and if they refer to mutable data one might
violate memory safety since holding multiple mutable references to shared data
is not allowed.

If you really want global mutable state, try using `static mut` or a global
`UnsafeCell`.
"##,

E0019: r##"
A function call isn't allowed in the const's initialization expression
because the expression's value must be known at compile-time.

Erroneous code example:

```compile_fail,E0019
#![feature(box_syntax)]

fn main() {
    struct MyOwned;

    static STATIC11: Box<MyOwned> = box MyOwned; // error!
}
```

Remember: you can't use a function call inside a const's initialization
expression! However, you can totally use it anywhere else:

```
enum Test {
    V1
}

impl Test {
    fn func(&self) -> i32 {
        12
    }
}

fn main() {
    const FOO: Test = Test::V1;

    FOO.func(); // here is good
    let x = FOO.func(); // or even here!
}
```
"##,

E0023: r##"
A pattern used to match against an enum variant must provide a sub-pattern for
each field of the enum variant. This error indicates that a pattern attempted to
extract an incorrect number of fields from a variant.

```
enum Fruit {
    Apple(String, String),
    Pear(u32),
}
```

Here the `Apple` variant has two fields, and should be matched against like so:

```
enum Fruit {
    Apple(String, String),
    Pear(u32),
}

let x = Fruit::Apple(String::new(), String::new());

// Correct.
match x {
    Fruit::Apple(a, b) => {},
    _ => {}
}
```

Matching with the wrong number of fields has no sensible interpretation:

```compile_fail,E0023
enum Fruit {
    Apple(String, String),
    Pear(u32),
}

let x = Fruit::Apple(String::new(), String::new());

// Incorrect.
match x {
    Fruit::Apple(a) => {},
    Fruit::Apple(a, b, c) => {},
}
```

Check how many fields the enum was declared with and ensure that your pattern
uses the same number.
"##,

E0025: r##"
Each field of a struct can only be bound once in a pattern. Erroneous code
example:

```compile_fail,E0025
struct Foo {
    a: u8,
    b: u8,
}

fn main(){
    let x = Foo { a:1, b:2 };

    let Foo { a: x, a: y } = x;
    // error: field `a` bound multiple times in the pattern
}
```

Each occurrence of a field name binds the value of that field, so to fix this
error you will have to remove or alter the duplicate uses of the field name.
Perhaps you misspelled another field name? Example:

```
struct Foo {
    a: u8,
    b: u8,
}

fn main(){
    let x = Foo { a:1, b:2 };

    let Foo { a: x, b: y } = x; // ok!
}
```
"##,

E0026: r##"
This error indicates that a struct pattern attempted to extract a non-existent
field from a struct. Struct fields are identified by the name used before the
colon `:` so struct patterns should resemble the declaration of the struct type
being matched.

```
// Correct matching.
struct Thing {
    x: u32,
    y: u32
}

let thing = Thing { x: 1, y: 2 };

match thing {
    Thing { x: xfield, y: yfield } => {}
}
```

If you are using shorthand field patterns but want to refer to the struct field
by a different name, you should rename it explicitly.

Change this:

```compile_fail,E0026
struct Thing {
    x: u32,
    y: u32
}

let thing = Thing { x: 0, y: 0 };

match thing {
    Thing { x, z } => {}
}
```

To this:

```
struct Thing {
    x: u32,
    y: u32
}

let thing = Thing { x: 0, y: 0 };

match thing {
    Thing { x, y: z } => {}
}
```
"##,

E0027: r##"
This error indicates that a pattern for a struct fails to specify a sub-pattern
for every one of the struct's fields. Ensure that each field from the struct's
definition is mentioned in the pattern, or use `..` to ignore unwanted fields.

For example:

```compile_fail,E0027
struct Dog {
    name: String,
    age: u32,
}

let d = Dog { name: "Rusty".to_string(), age: 8 };

// This is incorrect.
match d {
    Dog { age: x } => {}
}
```

This is correct (explicit):

```
struct Dog {
    name: String,
    age: u32,
}

let d = Dog { name: "Rusty".to_string(), age: 8 };

match d {
    Dog { name: ref n, age: x } => {}
}

// This is also correct (ignore unused fields).
match d {
    Dog { age: x, .. } => {}
}
```
"##,

E0029: r##"
In a match expression, only numbers and characters can be matched against a
range. This is because the compiler checks that the range is non-empty at
compile-time, and is unable to evaluate arbitrary comparison functions. If you
want to capture values of an orderable type between two end-points, you can use
a guard.

```compile_fail,E0029
let string = "salutations !";

// The ordering relation for strings cannot be evaluated at compile time,
// so this doesn't work:
match string {
    "hello" ..= "world" => {}
    _ => {}
}

// This is a more general version, using a guard:
match string {
    s if s >= "hello" && s <= "world" => {}
    _ => {}
}
```
"##,

E0030: r##"
When matching against a range, the compiler verifies that the range is
non-empty. Range patterns include both end-points, so this is equivalent to
requiring the start of the range to be less than or equal to the end of the
range.

Erroneous code example:

```compile_fail,E0030
match 5u32 {
    // This range is ok, albeit pointless.
    1 ..= 1 => {}
    // This range is empty, and the compiler can tell.
    1000 ..= 5 => {}
}
```
"##,

E0033: r##"
This error indicates that a pointer to a trait type cannot be implicitly
dereferenced by a pattern. Every trait defines a type, but because the
size of trait implementers isn't fixed, this type has no compile-time size.
Therefore, all accesses to trait types must be through pointers. If you
encounter this error you should try to avoid dereferencing the pointer.

```compile_fail,E0033
# trait SomeTrait { fn method_one(&self){} fn method_two(&self){} }
# impl<T> SomeTrait for T {}
let trait_obj: &SomeTrait = &"some_value";

// This tries to implicitly dereference to create an unsized local variable.
let &invalid = trait_obj;

// You can call methods without binding to the value being pointed at.
trait_obj.method_one();
trait_obj.method_two();
```

You can read more about trait objects in the [Trait Objects] section of the
Reference.

[Trait Objects]: https://doc.rust-lang.org/reference/types.html#trait-objects
"##,

E0034: r##"
The compiler doesn't know what method to call because more than one method
has the same prototype. Erroneous code example:

```compile_fail,E0034
struct Test;

trait Trait1 {
    fn foo();
}

trait Trait2 {
    fn foo();
}

impl Trait1 for Test { fn foo() {} }
impl Trait2 for Test { fn foo() {} }

fn main() {
    Test::foo() // error, which foo() to call?
}
```

To avoid this error, you have to keep only one of them and remove the others.
So let's take our example and fix it:

```
struct Test;

trait Trait1 {
    fn foo();
}

impl Trait1 for Test { fn foo() {} }

fn main() {
    Test::foo() // and now that's good!
}
```

However, a better solution would be using fully explicit naming of type and
trait:

```
struct Test;

trait Trait1 {
    fn foo();
}

trait Trait2 {
    fn foo();
}

impl Trait1 for Test { fn foo() {} }
impl Trait2 for Test { fn foo() {} }

fn main() {
    <Test as Trait1>::foo()
}
```

One last example:

```
trait F {
    fn m(&self);
}

trait G {
    fn m(&self);
}

struct X;

impl F for X { fn m(&self) { println!("I am F"); } }
impl G for X { fn m(&self) { println!("I am G"); } }

fn main() {
    let f = X;

    F::m(&f); // it displays "I am F"
    G::m(&f); // it displays "I am G"
}
```
"##,

E0038: r##"
Trait objects like `Box<Trait>` can only be constructed when certain
requirements are satisfied by the trait in question.

Trait objects are a form of dynamic dispatch and use a dynamically sized type
for the inner type. So, for a given trait `Trait`, when `Trait` is treated as a
type, as in `Box<Trait>`, the inner type is 'unsized'. In such cases the boxed
pointer is a 'fat pointer' that contains an extra pointer to a table of methods
(among other things) for dynamic dispatch. This design mandates some
restrictions on the types of traits that are allowed to be used in trait
objects, which are collectively termed as 'object safety' rules.

Attempting to create a trait object for a non object-safe trait will trigger
this error.

There are various rules:

### The trait cannot require `Self: Sized`

When `Trait` is treated as a type, the type does not implement the special
`Sized` trait, because the type does not have a known size at compile time and
can only be accessed behind a pointer. Thus, if we have a trait like the
following:

```
trait Foo where Self: Sized {

}
```

We cannot create an object of type `Box<Foo>` or `&Foo` since in this case
`Self` would not be `Sized`.

Generally, `Self: Sized` is used to indicate that the trait should not be used
as a trait object. If the trait comes from your own crate, consider removing
this restriction.

### Method references the `Self` type in its parameters or return type

This happens when a trait has a method like the following:

```
trait Trait {
    fn foo(&self) -> Self;
}

impl Trait for String {
    fn foo(&self) -> Self {
        "hi".to_owned()
    }
}

impl Trait for u8 {
    fn foo(&self) -> Self {
        1
    }
}
```

(Note that `&self` and `&mut self` are okay, it's additional `Self` types which
cause this problem.)

In such a case, the compiler cannot predict the return type of `foo()` in a
situation like the following:

```compile_fail
trait Trait {
    fn foo(&self) -> Self;
}

fn call_foo(x: Box<Trait>) {
    let y = x.foo(); // What type is y?
    // ...
}
```

If only some methods aren't object-safe, you can add a `where Self: Sized` bound
on them to mark them as explicitly unavailable to trait objects. The
functionality will still be available to all other implementers, including
`Box<Trait>` which is itself sized (assuming you `impl Trait for Box<Trait>`).

```
trait Trait {
    fn foo(&self) -> Self where Self: Sized;
    // more functions
}
```

Now, `foo()` can no longer be called on a trait object, but you will now be
allowed to make a trait object, and that will be able to call any object-safe
methods. With such a bound, one can still call `foo()` on types implementing
that trait that aren't behind trait objects.

### Method has generic type parameters

As mentioned before, trait objects contain pointers to method tables. So, if we
have:

```
trait Trait {
    fn foo(&self);
}

impl Trait for String {
    fn foo(&self) {
        // implementation 1
    }
}

impl Trait for u8 {
    fn foo(&self) {
        // implementation 2
    }
}
// ...
```

At compile time each implementation of `Trait` will produce a table containing
the various methods (and other items) related to the implementation.

This works fine, but when the method gains generic parameters, we can have a
problem.

Usually, generic parameters get _monomorphized_. For example, if I have

```
fn foo<T>(x: T) {
    // ...
}
```

The machine code for `foo::<u8>()`, `foo::<bool>()`, `foo::<String>()`, or any
other type substitution is different. Hence the compiler generates the
implementation on-demand. If you call `foo()` with a `bool` parameter, the
compiler will only generate code for `foo::<bool>()`. When we have additional
type parameters, the number of monomorphized implementations the compiler
generates does not grow drastically, since the compiler will only generate an
implementation if the function is called with unparametrized substitutions
(i.e., substitutions where none of the substituted types are themselves
parametrized).

However, with trait objects we have to make a table containing _every_ object
that implements the trait. Now, if it has type parameters, we need to add
implementations for every type that implements the trait, and there could
theoretically be an infinite number of types.

For example, with:

```
trait Trait {
    fn foo<T>(&self, on: T);
    // more methods
}

impl Trait for String {
    fn foo<T>(&self, on: T) {
        // implementation 1
    }
}

impl Trait for u8 {
    fn foo<T>(&self, on: T) {
        // implementation 2
    }
}

// 8 more implementations
```

Now, if we have the following code:

```compile_fail,E0038
# trait Trait { fn foo<T>(&self, on: T); }
# impl Trait for String { fn foo<T>(&self, on: T) {} }
# impl Trait for u8 { fn foo<T>(&self, on: T) {} }
# impl Trait for bool { fn foo<T>(&self, on: T) {} }
# // etc.
fn call_foo(thing: Box<Trait>) {
    thing.foo(true); // this could be any one of the 8 types above
    thing.foo(1);
    thing.foo("hello");
}
```

We don't just need to create a table of all implementations of all methods of
`Trait`, we need to create such a table, for each different type fed to
`foo()`. In this case this turns out to be (10 types implementing `Trait`)*(3
types being fed to `foo()`) = 30 implementations!

With real world traits these numbers can grow drastically.

To fix this, it is suggested to use a `where Self: Sized` bound similar to the
fix for the sub-error above if you do not intend to call the method with type
parameters:

```
trait Trait {
    fn foo<T>(&self, on: T) where Self: Sized;
    // more methods
}
```

If this is not an option, consider replacing the type parameter with another
trait object (e.g., if `T: OtherTrait`, use `on: Box<OtherTrait>`). If the
number of types you intend to feed to this method is limited, consider manually
listing out the methods of different types.

### Method has no receiver

Methods that do not take a `self` parameter can't be called since there won't be
a way to get a pointer to the method table for them.

```
trait Foo {
    fn foo() -> u8;
}
```

This could be called as `<Foo as Foo>::foo()`, which would not be able to pick
an implementation.

Adding a `Self: Sized` bound to these methods will generally make this compile.

```
trait Foo {
    fn foo() -> u8 where Self: Sized;
}
```

### The trait cannot contain associated constants

Just like static functions, associated constants aren't stored on the method
table. If the trait or any subtrait contain an associated constant, they cannot
be made into an object.

```compile_fail,E0038
trait Foo {
    const X: i32;
}

impl Foo {}
```

A simple workaround is to use a helper method instead:

```
trait Foo {
    fn x(&self) -> i32;
}
```

### The trait cannot use `Self` as a type parameter in the supertrait listing

This is similar to the second sub-error, but subtler. It happens in situations
like the following:

```compile_fail,E0038
trait Super<A: ?Sized> {}

trait Trait: Super<Self> {
}

struct Foo;

impl Super<Foo> for Foo{}

impl Trait for Foo {}

fn main() {
    let x: Box<dyn Trait>;
}
```

Here, the supertrait might have methods as follows:

```
trait Super<A: ?Sized> {
    fn get_a(&self) -> &A; // note that this is object safe!
}
```

If the trait `Trait` was deriving from something like `Super<String>` or
`Super<T>` (where `Foo` itself is `Foo<T>`), this is okay, because given a type
`get_a()` will definitely return an object of that type.

However, if it derives from `Super<Self>`, even though `Super` is object safe,
the method `get_a()` would return an object of unknown type when called on the
function. `Self` type parameters let us make object safe traits no longer safe,
so they are forbidden when specifying supertraits.

There's no easy fix for this, generally code will need to be refactored so that
you no longer need to derive from `Super<Self>`.
"##,

E0040: r##"
It is not allowed to manually call destructors in Rust. It is also not
necessary to do this since `drop` is called automatically whenever a value goes
out of scope.

Here's an example of this error:

```compile_fail,E0040
struct Foo {
    x: i32,
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("kaboom");
    }
}

fn main() {
    let mut x = Foo { x: -7 };
    x.drop(); // error: explicit use of destructor method
}
```
"##,

E0044: r##"
You cannot use type or const parameters on foreign items.
Example of erroneous code:

```compile_fail,E0044
extern { fn some_func<T>(x: T); }
```

To fix this, replace the generic parameter with the specializations that you
need:

```
extern { fn some_func_i32(x: i32); }
extern { fn some_func_i64(x: i64); }
```
"##,

E0045: r##"
Rust only supports variadic parameters for interoperability with C code in its
FFI. As such, variadic parameters can only be used with functions which are
using the C ABI. Examples of erroneous code:

```compile_fail
#![feature(unboxed_closures)]

extern "rust-call" { fn foo(x: u8, ...); }

// or

fn foo(x: u8, ...) {}
```

To fix such code, put them in an extern "C" block:

```
extern "C" {
    fn foo (x: u8, ...);
}
```
"##,

E0046: r##"
Items are missing in a trait implementation. Erroneous code example:

```compile_fail,E0046
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {}
// error: not all trait items implemented, missing: `foo`
```

When trying to make some type implement a trait `Foo`, you must, at minimum,
provide implementations for all of `Foo`'s required methods (meaning the
methods that do not have default implementations), as well as any required
trait items like associated types or constants. Example:

```
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {
    fn foo() {} // ok!
}
```
"##,

E0049: r##"
This error indicates that an attempted implementation of a trait method
has the wrong number of type or const parameters.

For example, the trait below has a method `foo` with a type parameter `T`,
but the implementation of `foo` for the type `Bar` is missing this parameter:

```compile_fail,E0049
trait Foo {
    fn foo<T: Default>(x: T) -> Self;
}

struct Bar;

// error: method `foo` has 0 type parameters but its trait declaration has 1
// type parameter
impl Foo for Bar {
    fn foo(x: bool) -> Self { Bar }
}
```
"##,

E0050: r##"
This error indicates that an attempted implementation of a trait method
has the wrong number of function parameters.

For example, the trait below has a method `foo` with two function parameters
(`&self` and `u8`), but the implementation of `foo` for the type `Bar` omits
the `u8` parameter:

```compile_fail,E0050
trait Foo {
    fn foo(&self, x: u8) -> bool;
}

struct Bar;

// error: method `foo` has 1 parameter but the declaration in trait `Foo::foo`
// has 2
impl Foo for Bar {
    fn foo(&self) -> bool { true }
}
```
"##,

E0053: r##"
The parameters of any trait method must match between a trait implementation
and the trait definition.

Here are a couple examples of this error:

```compile_fail,E0053
trait Foo {
    fn foo(x: u16);
    fn bar(&self);
}

struct Bar;

impl Foo for Bar {
    // error, expected u16, found i16
    fn foo(x: i16) { }

    // error, types differ in mutability
    fn bar(&mut self) { }
}
```
"##,

E0054: r##"
It is not allowed to cast to a bool. If you are trying to cast a numeric type
to a bool, you can compare it with zero instead:

```compile_fail,E0054
let x = 5;

// Not allowed, won't compile
let x_is_nonzero = x as bool;
```

```
let x = 5;

// Ok
let x_is_nonzero = x != 0;
```
"##,

E0055: r##"
During a method call, a value is automatically dereferenced as many times as
needed to make the value's type match the method's receiver. The catch is that
the compiler will only attempt to dereference a number of times up to the
recursion limit (which can be set via the `recursion_limit` attribute).

For a somewhat artificial example:

```compile_fail,E0055
#![recursion_limit="5"]

struct Foo;

impl Foo {
    fn foo(&self) {}
}

fn main() {
    let foo = Foo;
    let ref_foo = &&&&&Foo;

    // error, reached the recursion limit while auto-dereferencing `&&&&&Foo`
    ref_foo.foo();
}
```

One fix may be to increase the recursion limit. Note that it is possible to
create an infinite recursion of dereferencing, in which case the only fix is to
somehow break the recursion.
"##,

E0057: r##"
When invoking closures or other implementations of the function traits `Fn`,
`FnMut` or `FnOnce` using call notation, the number of parameters passed to the
function must match its definition.

An example using a closure:

```compile_fail,E0057
let f = |x| x * 3;
let a = f();        // invalid, too few parameters
let b = f(4);       // this works!
let c = f(2, 3);    // invalid, too many parameters
```

A generic function must be treated similarly:

```
fn foo<F: Fn()>(f: F) {
    f(); // this is valid, but f(3) would not work
}
```
"##,

E0059: r##"
The built-in function traits are generic over a tuple of the function arguments.
If one uses angle-bracket notation (`Fn<(T,), Output=U>`) instead of parentheses
(`Fn(T) -> U`) to denote the function trait, the type parameter should be a
tuple. Otherwise function call notation cannot be used and the trait will not be
implemented by closures.

The most likely source of this error is using angle-bracket notation without
wrapping the function argument type into a tuple, for example:

```compile_fail,E0059
#![feature(unboxed_closures)]

fn foo<F: Fn<i32>>(f: F) -> F::Output { f(3) }
```

It can be fixed by adjusting the trait bound like this:

```
#![feature(unboxed_closures)]

fn foo<F: Fn<(i32,)>>(f: F) -> F::Output { f(3) }
```

Note that `(T,)` always denotes the type of a 1-tuple containing an element of
type `T`. The comma is necessary for syntactic disambiguation.
"##,

E0060: r##"
External C functions are allowed to be variadic. However, a variadic function
takes a minimum number of arguments. For example, consider C's variadic `printf`
function:

```
use std::os::raw::{c_char, c_int};

extern "C" {
    fn printf(_: *const c_char, ...) -> c_int;
}
```

Using this declaration, it must be called with at least one argument, so
simply calling `printf()` is invalid. But the following uses are allowed:

```
# #![feature(static_nobundle)]
# use std::os::raw::{c_char, c_int};
# #[cfg_attr(all(windows, target_env = "msvc"),
#            link(name = "legacy_stdio_definitions", kind = "static-nobundle"))]
# extern "C" { fn printf(_: *const c_char, ...) -> c_int; }
# fn main() {
unsafe {
    use std::ffi::CString;

    let fmt = CString::new("test\n").unwrap();
    printf(fmt.as_ptr());

    let fmt = CString::new("number = %d\n").unwrap();
    printf(fmt.as_ptr(), 3);

    let fmt = CString::new("%d, %d\n").unwrap();
    printf(fmt.as_ptr(), 10, 5);
}
# }
```
"##,

E0061: r##"
The number of arguments passed to a function must match the number of arguments
specified in the function signature.

For example, a function like:

```
fn f(a: u16, b: &str) {}
```

Must always be called with exactly two arguments, e.g., `f(2, "test")`.

Note that Rust does not have a notion of optional function arguments or
variadic functions (except for its C-FFI).
"##,

E0062: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was specified more than once. Erroneous code
example:

```compile_fail,E0062
struct Foo {
    x: i32,
}

fn main() {
    let x = Foo {
                x: 0,
                x: 0, // error: field `x` specified more than once
            };
}
```

Each field should be specified exactly one time. Example:

```
struct Foo {
    x: i32,
}

fn main() {
    let x = Foo { x: 0 }; // ok!
}
```
"##,

E0063: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was not provided. Erroneous code example:

```compile_fail,E0063
struct Foo {
    x: i32,
    y: i32,
}

fn main() {
    let x = Foo { x: 0 }; // error: missing field: `y`
}
```

Each field should be specified exactly once. Example:

```
struct Foo {
    x: i32,
    y: i32,
}

fn main() {
    let x = Foo { x: 0, y: 0 }; // ok!
}
```
"##,

E0067: r##"
The left-hand side of a compound assignment expression must be a place
expression. A place expression represents a memory location and includes
item paths (ie, namespaced variables), dereferences, indexing expressions,
and field references.

Let's start with some erroneous code examples:

```compile_fail,E0067
use std::collections::LinkedList;

// Bad: assignment to non-place expression
LinkedList::new() += 1;

// ...

fn some_func(i: &mut i32) {
    i += 12; // Error : '+=' operation cannot be applied on a reference !
}
```

And now some working examples:

```
let mut i : i32 = 0;

i += 12; // Good !

// ...

fn some_func(i: &mut i32) {
    *i += 12; // Good !
}
```
"##,

E0069: r##"
The compiler found a function whose body contains a `return;` statement but
whose return type is not `()`. An example of this is:

```compile_fail,E0069
// error
fn foo() -> u8 {
    return;
}
```

Since `return;` is just like `return ();`, there is a mismatch between the
function's return type and the value being returned.
"##,

E0070: r##"
The left-hand side of an assignment operator must be a place expression. A
place expression represents a memory location and can be a variable (with
optional namespacing), a dereference, an indexing expression or a field
reference.

More details can be found in the [Expressions] section of the Reference.

[Expressions]: https://doc.rust-lang.org/reference/expressions.html#places-rvalues-and-temporaries

Now, we can go further. Here are some erroneous code examples:

```compile_fail,E0070
struct SomeStruct {
    x: i32,
    y: i32
}

const SOME_CONST : i32 = 12;

fn some_other_func() {}

fn some_function() {
    SOME_CONST = 14; // error : a constant value cannot be changed!
    1 = 3; // error : 1 isn't a valid place!
    some_other_func() = 4; // error : we cannot assign value to a function!
    SomeStruct.x = 12; // error : SomeStruct a structure name but it is used
                       // like a variable!
}
```

And now let's give working examples:

```
struct SomeStruct {
    x: i32,
    y: i32
}
let mut s = SomeStruct {x: 0, y: 0};

s.x = 3; // that's good !

// ...

fn some_func(x: &mut i32) {
    *x = 12; // that's good !
}
```
"##,

E0071: r##"
You tried to use structure-literal syntax to create an item that is
not a structure or enum variant.

Example of erroneous code:

```compile_fail,E0071
type U32 = u32;
let t = U32 { value: 4 }; // error: expected struct, variant or union type,
                          // found builtin type `u32`
```

To fix this, ensure that the name was correctly spelled, and that
the correct form of initializer was used.

For example, the code above can be fixed to:

```
enum Foo {
    FirstValue(i32)
}

fn main() {
    let u = Foo::FirstValue(0i32);

    let t = 4;
}
```
"##,

E0072: r##"
When defining a recursive struct or enum, any use of the type being defined
from inside the definition must occur behind a pointer (like `Box` or `&`).
This is because structs and enums must have a well-defined size, and without
the pointer, the size of the type would need to be unbounded.

Consider the following erroneous definition of a type for a list of bytes:

```compile_fail,E0072
// error, invalid recursive struct type
struct ListNode {
    head: u8,
    tail: Option<ListNode>,
}
```

This type cannot have a well-defined size, because it needs to be arbitrarily
large (since we would be able to nest `ListNode`s to any depth). Specifically,

```plain
size of `ListNode` = 1 byte for `head`
                   + 1 byte for the discriminant of the `Option`
                   + size of `ListNode`
```

One way to fix this is by wrapping `ListNode` in a `Box`, like so:

```
struct ListNode {
    head: u8,
    tail: Option<Box<ListNode>>,
}
```

This works because `Box` is a pointer, so its size is well-known.
"##,

E0073: r##"
#### Note: this error code is no longer emitted by the compiler.

You cannot define a struct (or enum) `Foo` that requires an instance of `Foo`
in order to make a new `Foo` value. This is because there would be no way a
first instance of `Foo` could be made to initialize another instance!

Here's an example of a struct that has this problem:

```
struct Foo { x: Box<Foo> } // error
```

One fix is to use `Option`, like so:

```
struct Foo { x: Option<Box<Foo>> }
```

Now it's possible to create at least one instance of `Foo`: `Foo { x: None }`.
"##,

E0074: r##"
#### Note: this error code is no longer emitted by the compiler.

When using the `#[simd]` attribute on a tuple struct, the components of the
tuple struct must all be of a concrete, nongeneric type so the compiler can
reason about how to use SIMD with them. This error will occur if the types
are generic.

This will cause an error:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Bad<T>(T, T, T);
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32);
```
"##,

E0075: r##"
The `#[simd]` attribute can only be applied to non empty tuple structs, because
it doesn't make sense to try to use SIMD operations when there are no values to
operate on.

This will cause an error:

```compile_fail,E0075
#![feature(repr_simd)]

#[repr(simd)]
struct Bad;
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32);
```
"##,

E0076: r##"
When using the `#[simd]` attribute to automatically use SIMD operations in tuple
struct, the types in the struct must all be of the same type, or the compiler
will trigger this error.

This will cause an error:

```compile_fail,E0076
#![feature(repr_simd)]

#[repr(simd)]
struct Bad(u16, u32, u32);
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32);
```
"##,

E0077: r##"
When using the `#[simd]` attribute on a tuple struct, the elements in the tuple
must be machine types so SIMD operations can be applied to them.

This will cause an error:

```compile_fail,E0077
#![feature(repr_simd)]

#[repr(simd)]
struct Bad(String);
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32);
```
"##,

E0080: r##"
This error indicates that the compiler was unable to sensibly evaluate a
constant expression that had to be evaluated. Attempting to divide by 0
or causing integer overflow are two ways to induce this error. For example:

```compile_fail,E0080
enum Enum {
    X = (1 << 500),
    Y = (1 / 0)
}
```

Ensure that the expressions given can be evaluated as the desired integer type.
See the FFI section of the Reference for more information about using a custom
integer type:

https://doc.rust-lang.org/reference.html#ffi-attributes
"##,

E0081: r##"
Enum discriminants are used to differentiate enum variants stored in memory.
This error indicates that the same value was used for two or more variants,
making them impossible to tell apart.

```compile_fail,E0081
// Bad.
enum Enum {
    P = 3,
    X = 3,
    Y = 5,
}
```

```
// Good.
enum Enum {
    P,
    X = 3,
    Y = 5,
}
```

Note that variants without a manually specified discriminant are numbered from
top to bottom starting from 0, so clashes can occur with seemingly unrelated
variants.

```compile_fail,E0081
enum Bad {
    X,
    Y = 0
}
```

Here `X` will have already been specified the discriminant 0 by the time `Y` is
encountered, so a conflict occurs.
"##,

E0084: r##"
An unsupported representation was attempted on a zero-variant enum.

Erroneous code example:

```compile_fail,E0084
#[repr(i32)]
enum NightsWatch {} // error: unsupported representation for zero-variant enum
```

It is impossible to define an integer type to be used to represent zero-variant
enum values because there are no zero-variant enum values. There is no way to
construct an instance of the following type using only safe code. So you have
two solutions. Either you add variants in your enum:

```
#[repr(i32)]
enum NightsWatch {
    JonSnow,
    Commander,
}
```

or you remove the integer represention of your enum:

```
enum NightsWatch {}
```
"##,

E0087: r##"
#### Note: this error code is no longer emitted by the compiler.

Too many type arguments were supplied for a function. For example:

```compile_fail,E0107
fn foo<T>() {}

fn main() {
    foo::<f64, bool>(); // error: wrong number of type arguments:
                        //        expected 1, found 2
}
```

The number of supplied arguments must exactly match the number of defined type
parameters.
"##,

E0088: r##"
#### Note: this error code is no longer emitted by the compiler.

You gave too many lifetime arguments. Erroneous code example:

```compile_fail,E0107
fn f() {}

fn main() {
    f::<'static>() // error: wrong number of lifetime arguments:
                   //        expected 0, found 1
}
```

Please check you give the right number of lifetime arguments. Example:

```
fn f() {}

fn main() {
    f() // ok!
}
```

It's also important to note that the Rust compiler can generally
determine the lifetime by itself. Example:

```
struct Foo {
    value: String
}

impl Foo {
    // it can be written like this
    fn get_value<'a>(&'a self) -> &'a str { &self.value }
    // but the compiler works fine with this too:
    fn without_lifetime(&self) -> &str { &self.value }
}

fn main() {
    let f = Foo { value: "hello".to_owned() };

    println!("{}", f.get_value());
    println!("{}", f.without_lifetime());
}
```
"##,

E0089: r##"
#### Note: this error code is no longer emitted by the compiler.

Too few type arguments were supplied for a function. For example:

```compile_fail,E0107
fn foo<T, U>() {}

fn main() {
    foo::<f64>(); // error: wrong number of type arguments: expected 2, found 1
}
```

Note that if a function takes multiple type arguments but you want the compiler
to infer some of them, you can use type placeholders:

```compile_fail,E0107
fn foo<T, U>(x: T) {}

fn main() {
    let x: bool = true;
    foo::<f64>(x);    // error: wrong number of type arguments:
                      //        expected 2, found 1
    foo::<_, f64>(x); // same as `foo::<bool, f64>(x)`
}
```
"##,

E0090: r##"
#### Note: this error code is no longer emitted by the compiler.

You gave too few lifetime arguments. Example:

```compile_fail,E0107
fn foo<'a: 'b, 'b: 'a>() {}

fn main() {
    foo::<'static>(); // error: wrong number of lifetime arguments:
                      //        expected 2, found 1
}
```

Please check you give the right number of lifetime arguments. Example:

```
fn foo<'a: 'b, 'b: 'a>() {}

fn main() {
    foo::<'static, 'static>();
}
```
"##,

E0091: r##"
You gave an unnecessary type or const parameter in a type alias. Erroneous
code example:

```compile_fail,E0091
type Foo<T> = u32; // error: type parameter `T` is unused
// or:
type Foo<A,B> = Box<A>; // error: type parameter `B` is unused
```

Please check you didn't write too many parameters. Example:

```
type Foo = u32; // ok!
type Foo2<A> = Box<A>; // ok!
```
"##,

E0092: r##"
You tried to declare an undefined atomic operation function.
Erroneous code example:

```compile_fail,E0092
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_foo(); // error: unrecognized atomic operation
                     //        function
}
```

Please check you didn't make a mistake in the function's name. All intrinsic
functions are defined in librustc_codegen_llvm/intrinsic.rs and in
libcore/intrinsics.rs in the Rust source code. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_fence(); // ok!
}
```
"##,

E0093: r##"
You declared an unknown intrinsic function. Erroneous code example:

```compile_fail,E0093
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn foo(); // error: unrecognized intrinsic function: `foo`
}

fn main() {
    unsafe {
        foo();
    }
}
```

Please check you didn't make a mistake in the function's name. All intrinsic
functions are defined in librustc_codegen_llvm/intrinsic.rs and in
libcore/intrinsics.rs in the Rust source code. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_fence(); // ok!
}

fn main() {
    unsafe {
        atomic_fence();
    }
}
```
"##,

E0094: r##"
You gave an invalid number of type parameters to an intrinsic function.
Erroneous code example:

```compile_fail,E0094
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T, U>() -> usize; // error: intrinsic has wrong number
                                 //        of type parameters
}
```

Please check that you provided the right number of type parameters
and verify with the function declaration in the Rust source code.
Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>() -> usize; // ok!
}
```
"##,

E0106: r##"
This error indicates that a lifetime is missing from a type. If it is an error
inside a function signature, the problem may be with failing to adhere to the
lifetime elision rules (see below).

Here are some simple examples of where you'll run into this error:

```compile_fail,E0106
struct Foo1 { x: &bool }
              // ^ expected lifetime parameter
struct Foo2<'a> { x: &'a bool } // correct

struct Bar1 { x: Foo2 }
              // ^^^^ expected lifetime parameter
struct Bar2<'a> { x: Foo2<'a> } // correct

enum Baz1 { A(u8), B(&bool), }
                  // ^ expected lifetime parameter
enum Baz2<'a> { A(u8), B(&'a bool), } // correct

type MyStr1 = &str;
           // ^ expected lifetime parameter
type MyStr2<'a> = &'a str; // correct
```

Lifetime elision is a special, limited kind of inference for lifetimes in
function signatures which allows you to leave out lifetimes in certain cases.
For more background on lifetime elision see [the book][book-le].

The lifetime elision rules require that any function signature with an elided
output lifetime must either have

 - exactly one input lifetime
 - or, multiple input lifetimes, but the function must also be a method with a
   `&self` or `&mut self` receiver

In the first case, the output lifetime is inferred to be the same as the unique
input lifetime. In the second case, the lifetime is instead inferred to be the
same as the lifetime on `&self` or `&mut self`.

Here are some examples of elision errors:

```compile_fail,E0106
// error, no input lifetimes
fn foo() -> &str { }

// error, `x` and `y` have distinct lifetimes inferred
fn bar(x: &str, y: &str) -> &str { }

// error, `y`'s lifetime is inferred to be distinct from `x`'s
fn baz<'a>(x: &'a str, y: &str) -> &str { }
```

[book-le]: https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#lifetime-elision
"##,

E0107: r##"
This error means that an incorrect number of generic arguments were provided:

```compile_fail,E0107
struct Foo<T> { x: T }

struct Bar { x: Foo }             // error: wrong number of type arguments:
                                  //        expected 1, found 0
struct Baz<S, T> { x: Foo<S, T> } // error: wrong number of type arguments:
                                  //        expected 1, found 2

fn foo<T, U>(x: T, y: U) {}

fn main() {
    let x: bool = true;
    foo::<bool>(x);                 // error: wrong number of type arguments:
                                    //        expected 2, found 1
    foo::<bool, i32, i32>(x, 2, 4); // error: wrong number of type arguments:
                                    //        expected 2, found 3
}

fn f() {}

fn main() {
    f::<'static>(); // error: wrong number of lifetime arguments:
                    //        expected 0, found 1
}
```

"##,

E0109: r##"
You tried to provide a generic argument to a type which doesn't need it.
Erroneous code example:

```compile_fail,E0109
type X = u32<i32>; // error: type arguments are not allowed for this type
type Y = bool<'static>; // error: lifetime parameters are not allowed on
                        //        this type
```

Check that you used the correct argument and that the definition is correct.

Example:

```
type X = u32; // ok!
type Y = bool; // ok!
```

Note that generic arguments for enum variant constructors go after the variant,
not after the enum. For example, you would write `Option::None::<u32>`,
rather than `Option::<u32>::None`.
"##,

E0110: r##"
#### Note: this error code is no longer emitted by the compiler.

You tried to provide a lifetime to a type which doesn't need it.
See `E0109` for more details.
"##,

E0116: r##"
You can only define an inherent implementation for a type in the same crate
where the type was defined. For example, an `impl` block as below is not allowed
since `Vec` is defined in the standard library:

```compile_fail,E0116
impl Vec<u8> { } // error
```

To fix this problem, you can do either of these things:

 - define a trait that has the desired associated functions/types/constants and
   implement the trait for the type in question
 - define a new type wrapping the type and define an implementation on the new
   type

Note that using the `type` keyword does not work here because `type` only
introduces a type alias:

```compile_fail,E0116
type Bytes = Vec<u8>;

impl Bytes { } // error, same as above
```
"##,

E0117: r##"
This error indicates a violation of one of Rust's orphan rules for trait
implementations. The rule prohibits any implementation of a foreign trait (a
trait defined in another crate) where

 - the type that is implementing the trait is foreign
 - all of the parameters being passed to the trait (if there are any) are also
   foreign.

Here's one example of this error:

```compile_fail,E0117
impl Drop for u32 {}
```

To avoid this kind of error, ensure that at least one local type is referenced
by the `impl`:

```
pub struct Foo; // you define your type in your crate

impl Drop for Foo { // and you can implement the trait on it!
    // code of trait implementation here
#   fn drop(&mut self) { }
}

impl From<Foo> for i32 { // or you use a type from your crate as
                         // a type parameter
    fn from(i: Foo) -> i32 {
        0
    }
}
```

Alternatively, define a trait locally and implement that instead:

```
trait Bar {
    fn get(&self) -> usize;
}

impl Bar for u32 {
    fn get(&self) -> usize { 0 }
}
```

For information on the design of the orphan rules, see [RFC 1023].

[RFC 1023]: https://github.com/rust-lang/rfcs/blob/master/text/1023-rebalancing-coherence.md
"##,

E0118: r##"
You're trying to write an inherent implementation for something which isn't a
struct nor an enum. Erroneous code example:

```compile_fail,E0118
impl (u8, u8) { // error: no base type found for inherent implementation
    fn get_state(&self) -> String {
        // ...
    }
}
```

To fix this error, please implement a trait on the type or wrap it in a struct.
Example:

```
// we create a trait here
trait LiveLongAndProsper {
    fn get_state(&self) -> String;
}

// and now you can implement it on (u8, u8)
impl LiveLongAndProsper for (u8, u8) {
    fn get_state(&self) -> String {
        "He's dead, Jim!".to_owned()
    }
}
```

Alternatively, you can create a newtype. A newtype is a wrapping tuple-struct.
For example, `NewType` is a newtype over `Foo` in `struct NewType(Foo)`.
Example:

```
struct TypeWrapper((u8, u8));

impl TypeWrapper {
    fn get_state(&self) -> String {
        "Fascinating!".to_owned()
    }
}
```
"##,

E0119: r##"
There are conflicting trait implementations for the same type.
Example of erroneous code:

```compile_fail,E0119
trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

struct Foo {
    value: usize
}

impl MyTrait for Foo { // error: conflicting implementations of trait
                       //        `MyTrait` for type `Foo`
    fn get(&self) -> usize { self.value }
}
```

When looking for the implementation for the trait, the compiler finds
both the `impl<T> MyTrait for T` where T is all types and the `impl
MyTrait for Foo`. Since a trait cannot be implemented multiple times,
this is an error. So, when you write:

```
trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}
```

This makes the trait implemented on all types in the scope. So if you
try to implement it on another one after that, the implementations will
conflict. Example:

```
trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

struct Foo;

fn main() {
    let f = Foo;

    f.get(); // the trait is implemented so we can use it
}
```
"##,

E0120: r##"
An attempt was made to implement Drop on a trait, which is not allowed: only
structs and enums can implement Drop. An example causing this error:

```compile_fail,E0120
trait MyTrait {}

impl Drop for MyTrait {
    fn drop(&mut self) {}
}
```

A workaround for this problem is to wrap the trait up in a struct, and implement
Drop on that. An example is shown below:

```
trait MyTrait {}
struct MyWrapper<T: MyTrait> { foo: T }

impl <T: MyTrait> Drop for MyWrapper<T> {
    fn drop(&mut self) {}
}

```

Alternatively, wrapping trait objects requires something like the following:

```
trait MyTrait {}

//or Box<MyTrait>, if you wanted an owned trait object
struct MyWrapper<'a> { foo: &'a MyTrait }

impl <'a> Drop for MyWrapper<'a> {
    fn drop(&mut self) {}
}
```
"##,

E0121: r##"
In order to be consistent with Rust's lack of global type inference,
type and const placeholders are disallowed by design in item signatures.

Examples of this error include:

```compile_fail,E0121
fn foo() -> _ { 5 } // error, explicitly write out the return type instead

static BAR: _ = "test"; // error, explicitly write out the type instead
```
"##,

E0124: r##"
You declared two fields of a struct with the same name. Erroneous code
example:

```compile_fail,E0124
struct Foo {
    field1: i32,
    field1: i32, // error: field is already declared
}
```

Please verify that the field names have been correctly spelled. Example:

```
struct Foo {
    field1: i32,
    field2: i32, // ok!
}
```
"##,

E0128: r##"
Type parameter defaults can only use parameters that occur before them.
Erroneous code example:

```compile_fail,E0128
struct Foo<T = U, U = ()> {
    field1: T,
    field2: U,
}
// error: type parameters with a default cannot use forward declared
// identifiers
```

Since type parameters are evaluated in-order, you may be able to fix this issue
by doing:

```
struct Foo<U = (), T = U> {
    field1: T,
    field2: U,
}
```

Please also verify that this wasn't because of a name-clash and rename the type
parameter if so.
"##,

E0130: r##"
You declared a pattern as an argument in a foreign function declaration.

Erroneous code example:

```compile_fail
extern {
    fn foo((a, b): (u32, u32)); // error: patterns aren't allowed in foreign
                                //        function declarations
}
```

Please replace the pattern argument with a regular one. Example:

```
struct SomeStruct {
    a: u32,
    b: u32,
}

extern {
    fn foo(s: SomeStruct); // ok!
}
```

Or:

```
extern {
    fn foo(a: (u32, u32)); // ok!
}
```
"##,

E0131: r##"
It is not possible to define `main` with generic parameters.
When `main` is present, it must take no arguments and return `()`.
Erroneous code example:

```compile_fail,E0131
fn main<T>() { // error: main function is not allowed to have generic parameters
}
```
"##,

E0132: r##"
A function with the `start` attribute was declared with type parameters.

Erroneous code example:

```compile_fail,E0132
#![feature(start)]

#[start]
fn f<T>() {}
```

It is not possible to declare type parameters on a function that has the `start`
attribute. Such a function must have the following type signature (for more
information, view [the unstable book][1]):

[1]: https://doc.rust-lang.org/unstable-book/language-features/lang-items.html#writing-an-executable-without-stdlib

```
# let _:
fn(isize, *const *const u8) -> isize;
```

Example:

```
#![feature(start)]

#[start]
fn my_start(argc: isize, argv: *const *const u8) -> isize {
    0
}
```
"##,

E0133: r##"
Unsafe code was used outside of an unsafe function or block.

Erroneous code example:

```compile_fail,E0133
unsafe fn f() { return; } // This is the unsafe code

fn main() {
    f(); // error: call to unsafe function requires unsafe function or block
}
```

Using unsafe functionality is potentially dangerous and disallowed by safety
checks. Examples:

* Dereferencing raw pointers
* Calling functions via FFI
* Calling functions marked unsafe

These safety checks can be relaxed for a section of the code by wrapping the
unsafe instructions with an `unsafe` block. For instance:

```
unsafe fn f() { return; }

fn main() {
    unsafe { f(); } // ok!
}
```

See also https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
"##,

E0136: r##"
A binary can only have one entry point, and by default that entry point is the
function `main()`. If there are multiple such functions, please rename one.

Erroneous code example:

```compile_fail,E0136
fn main() {
    // ...
}

// ...

fn main() { // error!
    // ...
}
```
"##,

E0137: r##"
More than one function was declared with the `#[main]` attribute.

Erroneous code example:

```compile_fail,E0137
#![feature(main)]

#[main]
fn foo() {}

#[main]
fn f() {} // error: multiple functions with a `#[main]` attribute
```

This error indicates that the compiler found multiple functions with the
`#[main]` attribute. This is an error because there must be a unique entry
point into a Rust program. Example:

```
#![feature(main)]

#[main]
fn f() {} // ok!
```
"##,

E0138: r##"
More than one function was declared with the `#[start]` attribute.

Erroneous code example:

```compile_fail,E0138
#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize {}

#[start]
fn f(argc: isize, argv: *const *const u8) -> isize {}
// error: multiple 'start' functions
```

This error indicates that the compiler found multiple functions with the
`#[start]` attribute. This is an error because there must be a unique entry
point into a Rust program. Example:

```
#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize { 0 } // ok!
```
"##,

E0139: r##"
#### Note: this error code is no longer emitted by the compiler.

There are various restrictions on transmuting between types in Rust; for example
types being transmuted must have the same size. To apply all these restrictions,
the compiler must know the exact types that may be transmuted. When type
parameters are involved, this cannot always be done.

So, for example, the following is not allowed:

```
use std::mem::transmute;

struct Foo<T>(Vec<T>);

fn foo<T>(x: Vec<T>) {
    // we are transmuting between Vec<T> and Foo<F> here
    let y: Foo<T> = unsafe { transmute(x) };
    // do something with y
}
```

In this specific case there's a good chance that the transmute is harmless (but
this is not guaranteed by Rust). However, when alignment and enum optimizations
come into the picture, it's quite likely that the sizes may or may not match
with different type parameter substitutions. It's not possible to check this for
_all_ possible types, so `transmute()` simply only accepts types without any
unsubstituted type parameters.

If you need this, there's a good chance you're doing something wrong. Keep in
mind that Rust doesn't guarantee much about the layout of different structs
(even two structs with identical declarations may have different layouts). If
there is a solution that avoids the transmute entirely, try it instead.

If it's possible, hand-monomorphize the code by writing the function for each
possible type substitution. It's possible to use traits to do this cleanly,
for example:

```
use std::mem::transmute;

struct Foo<T>(Vec<T>);

trait MyTransmutableType: Sized {
    fn transmute(_: Vec<Self>) -> Foo<Self>;
}

impl MyTransmutableType for u8 {
    fn transmute(x: Vec<u8>) -> Foo<u8> {
        unsafe { transmute(x) }
    }
}

impl MyTransmutableType for String {
    fn transmute(x: Vec<String>) -> Foo<String> {
        unsafe { transmute(x) }
    }
}

// ... more impls for the types you intend to transmute

fn foo<T: MyTransmutableType>(x: Vec<T>) {
    let y: Foo<T> = <T as MyTransmutableType>::transmute(x);
    // do something with y
}
```

Each impl will be checked for a size match in the transmute as usual, and since
there are no unbound type parameters involved, this should compile unless there
is a size mismatch in one of the impls.

It is also possible to manually transmute:

```
# use std::ptr;
# let v = Some("value");
# type SomeType = &'static [u8];
unsafe {
    ptr::read(&v as *const _ as *const SomeType) // `v` transmuted to `SomeType`
}
# ;
```

Note that this does not move `v` (unlike `transmute`), and may need a
call to `mem::forget(v)` in case you want to avoid destructors being called.
"##,

E0152: r##"
A lang item was redefined.

Erroneous code example:

```compile_fail,E0152
#![feature(lang_items)]

#[lang = "arc"]
struct Foo; // error: duplicate lang item found: `arc`
```

Lang items are already implemented in the standard library. Unless you are
writing a free-standing application (e.g., a kernel), you do not need to provide
them yourself.

You can build a free-standing crate by adding `#![no_std]` to the crate
attributes:

```ignore (only-for-syntax-highlight)
#![no_std]
```

See also the [unstable book][1].

[1]: https://doc.rust-lang.org/unstable-book/language-features/lang-items.html#writing-an-executable-without-stdlib
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

E0158: r##"
An associated const has been referenced in a pattern.

Erroneous code example:

```compile_fail,E0158
enum EFoo { A, B, C, D }

trait Foo {
    const X: EFoo;
}

fn test<A: Foo>(arg: EFoo) {
    match arg {
        A::X => { // error!
            println!("A::X");
        }
    }
}
```

`const` and `static` mean different things. A `const` is a compile-time
constant, an alias for a literal value. This property means you can match it
directly within a pattern.

The `static` keyword, on the other hand, guarantees a fixed location in memory.
This does not always mean that the value is constant. For example, a global
mutex can be declared `static` as well.

If you want to match against a `static`, consider using a guard instead:

```
static FORTY_TWO: i32 = 42;

match Some(42) {
    Some(x) if x == FORTY_TWO => {}
    _ => {}
}
```
"##,

E0161: r##"
A value was moved. However, its size was not known at compile time, and only
values of a known size can be moved.

Erroneous code example:

```compile_fail,E0161
#![feature(box_syntax)]

fn main() {
    let array: &[isize] = &[1, 2, 3];
    let _x: Box<[isize]> = box *array;
    // error: cannot move a value of type [isize]: the size of [isize] cannot
    //        be statically determined
}
```

In Rust, you can only move a value when its size is known at compile time.

To work around this restriction, consider "hiding" the value behind a reference:
either `&x` or `&mut x`. Since a reference has a fixed size, this lets you move
it around as usual. Example:

```
#![feature(box_syntax)]

fn main() {
    let array: &[isize] = &[1, 2, 3];
    let _x: Box<&[isize]> = box array; // ok!
}
```
"##,

E0162: r##"
#### Note: this error code is no longer emitted by the compiler.

An if-let pattern attempts to match the pattern, and enters the body if the
match was successful. If the match is irrefutable (when it cannot fail to
match), use a regular `let`-binding instead. For instance:

```
struct Irrefutable(i32);
let irr = Irrefutable(0);

// This fails to compile because the match is irrefutable.
if let Irrefutable(x) = irr {
    // This body will always be executed.
    // ...
}
```

Try this instead:

```
struct Irrefutable(i32);
let irr = Irrefutable(0);

let Irrefutable(x) = irr;
println!("{}", x);
```
"##,

E0164: r##"
This error means that an attempt was made to match a struct type enum
variant as a non-struct type:

```compile_fail,E0164
enum Foo { B { i: u32 } }

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B(i) => i, // error E0164
    }
}
```

Try using `{}` instead:

```
enum Foo { B { i: u32 } }

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B{i} => i,
    }
}
```
"##,

E0165: r##"
#### Note: this error code is no longer emitted by the compiler.

A while-let pattern attempts to match the pattern, and enters the body if the
match was successful. If the match is irrefutable (when it cannot fail to
match), use a regular `let`-binding inside a `loop` instead. For instance:

```no_run
struct Irrefutable(i32);
let irr = Irrefutable(0);

// This fails to compile because the match is irrefutable.
while let Irrefutable(x) = irr {
    // ...
}
```

Try this instead:

```no_run
struct Irrefutable(i32);
let irr = Irrefutable(0);

loop {
    let Irrefutable(x) = irr;
    // ...
}
```
"##,

E0170: r##"
Enum variants are qualified by default. For example, given this type:

```
enum Method {
    GET,
    POST,
}
```

You would match it using:

```
enum Method {
    GET,
    POST,
}

let m = Method::GET;

match m {
    Method::GET => {},
    Method::POST => {},
}
```

If you don't qualify the names, the code will bind new variables named "GET" and
"POST" instead. This behavior is likely not what you want, so `rustc` warns when
that happens.

Qualified names are good practice, and most code works well with them. But if
you prefer them unqualified, you can import the variants into scope:

```
use Method::*;
enum Method { GET, POST }
# fn main() {}
```

If you want others to be able to import variants from your module directly, use
`pub use`:

```
pub use Method::*;
pub enum Method { GET, POST }
# fn main() {}
```
"##,

E0178: r##"
In types, the `+` type operator has low precedence, so it is often necessary
to use parentheses.

For example:

```compile_fail,E0178
trait Foo {}

struct Bar<'a> {
    w: &'a Foo + Copy,   // error, use &'a (Foo + Copy)
    x: &'a Foo + 'a,     // error, use &'a (Foo + 'a)
    y: &'a mut Foo + 'a, // error, use &'a mut (Foo + 'a)
    z: fn() -> Foo + 'a, // error, use fn() -> (Foo + 'a)
}
```

More details can be found in [RFC 438].

[RFC 438]: https://github.com/rust-lang/rfcs/pull/438
"##,

E0184: r##"
Explicitly implementing both Drop and Copy for a type is currently disallowed.
This feature can make some sense in theory, but the current implementation is
incorrect and can lead to memory unsafety (see [issue #20126][iss20126]), so
it has been disabled for now.

[iss20126]: https://github.com/rust-lang/rust/issues/20126
"##,

E0185: r##"
An associated function for a trait was defined to be static, but an
implementation of the trait declared the same function to be a method (i.e., to
take a `self` parameter).

Here's an example of this error:

```compile_fail,E0185
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {
    // error, method `foo` has a `&self` declaration in the impl, but not in
    // the trait
    fn foo(&self) {}
}
```
"##,

E0186: r##"
An associated function for a trait was defined to be a method (i.e., to take a
`self` parameter), but an implementation of the trait declared the same function
to be static.

Here's an example of this error:

```compile_fail,E0186
trait Foo {
    fn foo(&self);
}

struct Bar;

impl Foo for Bar {
    // error, method `foo` has a `&self` declaration in the trait, but not in
    // the impl
    fn foo() {}
}
```
"##,

E0191: r##"
Trait objects need to have all associated types specified. Erroneous code
example:

```compile_fail,E0191
trait Trait {
    type Bar;
}

type Foo = Trait; // error: the value of the associated type `Bar` (from
                  //        the trait `Trait`) must be specified
```

Please verify you specified all associated types of the trait and that you
used the right trait. Example:

```
trait Trait {
    type Bar;
}

type Foo = Trait<Bar=i32>; // ok!
```
"##,

E0192: r##"
Negative impls are only allowed for auto traits. For more
information see the [opt-in builtin traits RFC][RFC 19].

[RFC 19]: https://github.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md
"##,

E0193: r##"
#### Note: this error code is no longer emitted by the compiler.

`where` clauses must use generic type parameters: it does not make sense to use
them otherwise. An example causing this error:

```
trait Foo {
    fn bar(&self);
}

#[derive(Copy,Clone)]
struct Wrapper<T> {
    Wrapped: T
}

impl Foo for Wrapper<u32> where Wrapper<u32>: Clone {
    fn bar(&self) { }
}
```

This use of a `where` clause is strange - a more common usage would look
something like the following:

```
trait Foo {
    fn bar(&self);
}

#[derive(Copy,Clone)]
struct Wrapper<T> {
    Wrapped: T
}
impl <T> Foo for Wrapper<T> where Wrapper<T>: Clone {
    fn bar(&self) { }
}
```

Here, we're saying that the implementation exists on Wrapper only when the
wrapped type `T` implements `Clone`. The `where` clause is important because
some types will not implement `Clone`, and thus will not get this method.

In our erroneous example, however, we're referencing a single concrete type.
Since we know for certain that `Wrapper<u32>` implements `Clone`, there's no
reason to also specify it in a `where` clause.
"##,

E0195: r##"
Your method's lifetime parameters do not match the trait declaration.
Erroneous code example:

```compile_fail,E0195
trait Trait {
    fn bar<'a,'b:'a>(x: &'a str, y: &'b str);
}

struct Foo;

impl Trait for Foo {
    fn bar<'a,'b>(x: &'a str, y: &'b str) {
    // error: lifetime parameters or bounds on method `bar`
    // do not match the trait declaration
    }
}
```

The lifetime constraint `'b` for bar() implementation does not match the
trait declaration. Ensure lifetime declarations match exactly in both trait
declaration and implementation. Example:

```
trait Trait {
    fn t<'a,'b:'a>(x: &'a str, y: &'b str);
}

struct Foo;

impl Trait for Foo {
    fn t<'a,'b:'a>(x: &'a str, y: &'b str) { // ok!
    }
}
```
"##,

E0197: r##"
Inherent implementations (one that do not implement a trait but provide
methods associated with a type) are always safe because they are not
implementing an unsafe trait. Removing the `unsafe` keyword from the inherent
implementation will resolve this error.

```compile_fail,E0197
struct Foo;

// this will cause this error
unsafe impl Foo { }
// converting it to this will fix it
impl Foo { }
```
"##,

E0198: r##"
A negative implementation is one that excludes a type from implementing a
particular trait. Not being able to use a trait is always a safe operation,
so negative implementations are always safe and never need to be marked as
unsafe.

```compile_fail
#![feature(optin_builtin_traits)]

struct Foo;

// unsafe is unnecessary
unsafe impl !Clone for Foo { }
```

This will compile:

```ignore (ignore auto_trait future compatibility warning)
#![feature(optin_builtin_traits)]

struct Foo;

auto trait Enterprise {}

impl !Enterprise for Foo { }
```

Please note that negative impls are only allowed for auto traits.
"##,

E0199: r##"
Safe traits should not have unsafe implementations, therefore marking an
implementation for a safe trait unsafe will cause a compiler error. Removing
the unsafe marker on the trait noted in the error will resolve this problem.

```compile_fail,E0199
struct Foo;

trait Bar { }

// this won't compile because Bar is safe
unsafe impl Bar for Foo { }
// this will compile
impl Bar for Foo { }
```
"##,

E0200: r##"
Unsafe traits must have unsafe implementations. This error occurs when an
implementation for an unsafe trait isn't marked as unsafe. This may be resolved
by marking the unsafe implementation as unsafe.

```compile_fail,E0200
struct Foo;

unsafe trait Bar { }

// this won't compile because Bar is unsafe and impl isn't unsafe
impl Bar for Foo { }
// this will compile
unsafe impl Bar for Foo { }
```
"##,

E0201: r##"
It is an error to define two associated items (like methods, associated types,
associated functions, etc.) with the same identifier.

For example:

```compile_fail,E0201
struct Foo(u8);

impl Foo {
    fn bar(&self) -> bool { self.0 > 5 }
    fn bar() {} // error: duplicate associated function
}

trait Baz {
    type Quux;
    fn baz(&self) -> bool;
}

impl Baz for Foo {
    type Quux = u32;

    fn baz(&self) -> bool { true }

    // error: duplicate method
    fn baz(&self) -> bool { self.0 > 5 }

    // error: duplicate associated type
    type Quux = u32;
}
```

Note, however, that items with the same name are allowed for inherent `impl`
blocks that don't overlap:

```
struct Foo<T>(T);

impl Foo<u8> {
    fn bar(&self) -> bool { self.0 > 5 }
}

impl Foo<bool> {
    fn bar(&self) -> bool { self.0 }
}
```
"##,

E0202: r##"
Inherent associated types were part of [RFC 195] but are not yet implemented.
See [the tracking issue][iss8995] for the status of this implementation.

[RFC 195]: https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md
[iss8995]: https://github.com/rust-lang/rust/issues/8995
"##,

E0204: r##"
An attempt to implement the `Copy` trait for a struct failed because one of the
fields does not implement `Copy`. To fix this, you must implement `Copy` for the
mentioned field. Note that this may not be possible, as in the example of

```compile_fail,E0204
struct Foo {
    foo : Vec<u32>,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```compile_fail,E0204
#[derive(Copy)]
struct Foo<'a> {
    ty: &'a mut bool,
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is always `Copy`).
"##,

E0205: r##"
#### Note: this error code is no longer emitted by the compiler.

An attempt to implement the `Copy` trait for an enum failed because one of the
variants does not implement `Copy`. To fix this, you must implement `Copy` for
the mentioned variant. Note that this may not be possible, as in the example of

```compile_fail,E0204
enum Foo {
    Bar(Vec<u32>),
    Baz,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```compile_fail,E0204
#[derive(Copy)]
enum Foo<'a> {
    Bar(&'a mut bool),
    Baz,
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is always `Copy`).
"##,

E0206: r##"
You can only implement `Copy` for a struct or enum. Both of the following
examples will fail, because neither `[u8; 256]` nor `&'static mut Bar`
(mutable reference to `Bar`) is a struct or enum:

```compile_fail,E0206
type Foo = [u8; 256];
impl Copy for Foo { } // error

#[derive(Copy, Clone)]
struct Bar;
impl Copy for &'static mut Bar { } // error
```
"##,

E0207: r##"
Any type parameter or lifetime parameter of an `impl` must meet at least one of
the following criteria:

 - it appears in the _implementing type_ of the impl, e.g. `impl<T> Foo<T>`
 - for a trait impl, it appears in the _implemented trait_, e.g.
   `impl<T> SomeTrait<T> for Foo`
 - it is bound as an associated type, e.g. `impl<T, U> SomeTrait for T
   where T: AnotherTrait<AssocType=U>`

### Error example 1

Suppose we have a struct `Foo` and we would like to define some methods for it.
The following definition leads to a compiler error:

```compile_fail,E0207
struct Foo;

impl<T: Default> Foo {
// error: the type parameter `T` is not constrained by the impl trait, self
// type, or predicates [E0207]
    fn get(&self) -> T {
        <T as Default>::default()
    }
}
```

The problem is that the parameter `T` does not appear in the implementing type
(`Foo`) of the impl. In this case, we can fix the error by moving the type
parameter from the `impl` to the method `get`:


```
struct Foo;

// Move the type parameter from the impl to the method
impl Foo {
    fn get<T: Default>(&self) -> T {
        <T as Default>::default()
    }
}
```

### Error example 2

As another example, suppose we have a `Maker` trait and want to establish a
type `FooMaker` that makes `Foo`s:

```compile_fail,E0207
trait Maker {
    type Item;
    fn make(&mut self) -> Self::Item;
}

struct Foo<T> {
    foo: T
}

struct FooMaker;

impl<T: Default> Maker for FooMaker {
// error: the type parameter `T` is not constrained by the impl trait, self
// type, or predicates [E0207]
    type Item = Foo<T>;

    fn make(&mut self) -> Foo<T> {
        Foo { foo: <T as Default>::default() }
    }
}
```

This fails to compile because `T` does not appear in the trait or in the
implementing type.

One way to work around this is to introduce a phantom type parameter into
`FooMaker`, like so:

```
use std::marker::PhantomData;

trait Maker {
    type Item;
    fn make(&mut self) -> Self::Item;
}

struct Foo<T> {
    foo: T
}

// Add a type parameter to `FooMaker`
struct FooMaker<T> {
    phantom: PhantomData<T>,
}

impl<T: Default> Maker for FooMaker<T> {
    type Item = Foo<T>;

    fn make(&mut self) -> Foo<T> {
        Foo {
            foo: <T as Default>::default(),
        }
    }
}
```

Another way is to do away with the associated type in `Maker` and use an input
type parameter instead:

```
// Use a type parameter instead of an associated type here
trait Maker<Item> {
    fn make(&mut self) -> Item;
}

struct Foo<T> {
    foo: T
}

struct FooMaker;

impl<T: Default> Maker<Foo<T>> for FooMaker {
    fn make(&mut self) -> Foo<T> {
        Foo { foo: <T as Default>::default() }
    }
}
```

### Additional information

For more information, please see [RFC 447].

[RFC 447]: https://github.com/rust-lang/rfcs/blob/master/text/0447-no-unused-impl-parameters.md
"##,

E0210: r##"
This error indicates a violation of one of Rust's orphan rules for trait
implementations. The rule concerns the use of type parameters in an
implementation of a foreign trait (a trait defined in another crate), and
states that type parameters must be "covered" by a local type. To understand
what this means, it is perhaps easiest to consider a few examples.

If `ForeignTrait` is a trait defined in some external crate `foo`, then the
following trait `impl` is an error:

```compile_fail,E0210
# #[cfg(for_demonstration_only)]
extern crate foo;
# #[cfg(for_demonstration_only)]
use foo::ForeignTrait;
# use std::panic::UnwindSafe as ForeignTrait;

impl<T> ForeignTrait for T { } // error
# fn main() {}
```

To work around this, it can be covered with a local type, `MyType`:

```
# use std::panic::UnwindSafe as ForeignTrait;
struct MyType<T>(T);
impl<T> ForeignTrait for MyType<T> { } // Ok
```

Please note that a type alias is not sufficient.

For another example of an error, suppose there's another trait defined in `foo`
named `ForeignTrait2` that takes two type parameters. Then this `impl` results
in the same rule violation:

```ignore (cannot-doctest-multicrate-project)
struct MyType2;
impl<T> ForeignTrait2<T, MyType<T>> for MyType2 { } // error
```

The reason for this is that there are two appearances of type parameter `T` in
the `impl` header, both as parameters for `ForeignTrait2`. The first appearance
is uncovered, and so runs afoul of the orphan rule.

Consider one more example:

```ignore (cannot-doctest-multicrate-project)
impl<T> ForeignTrait2<MyType<T>, T> for MyType2 { } // Ok
```

This only differs from the previous `impl` in that the parameters `T` and
`MyType<T>` for `ForeignTrait2` have been swapped. This example does *not*
violate the orphan rule; it is permitted.

To see why that last example was allowed, you need to understand the general
rule. Unfortunately this rule is a bit tricky to state. Consider an `impl`:

```ignore (only-for-syntax-highlight)
impl<P1, ..., Pm> ForeignTrait<T1, ..., Tn> for T0 { ... }
```

where `P1, ..., Pm` are the type parameters of the `impl` and `T0, ..., Tn`
are types. One of the types `T0, ..., Tn` must be a local type (this is another
orphan rule, see the explanation for E0117). Let `i` be the smallest integer
such that `Ti` is a local type. Then no type parameter can appear in any of the
`Tj` for `j < i`.

For information on the design of the orphan rules, see [RFC 1023].

[RFC 1023]: https://github.com/rust-lang/rfcs/blob/master/text/1023-rebalancing-coherence.md
"##,

E0211: r##"
#### Note: this error code is no longer emitted by the compiler.

You used a function or type which doesn't fit the requirements for where it was
used. Erroneous code examples:

```compile_fail
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>(); // error: intrinsic has wrong type
}

// or:

fn main() -> i32 { 0 }
// error: main function expects type: `fn() {main}`: expected (), found i32

// or:

let x = 1u8;
match x {
    0u8..=3i8 => (),
    // error: mismatched types in range: expected u8, found i8
    _ => ()
}

// or:

use std::rc::Rc;
struct Foo;

impl Foo {
    fn x(self: Rc<Foo>) {}
    // error: mismatched self type: expected `Foo`: expected struct
    //        `Foo`, found struct `alloc::rc::Rc`
}
```

For the first code example, please check the function definition. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>() -> usize; // ok!
}
```

The second case example is a bit particular: the main function must always
have this definition:

```compile_fail
fn main();
```

They never take parameters and never return types.

For the third example, when you match, all patterns must have the same type
as the type you're matching on. Example:

```
let x = 1u8;

match x {
    0u8..=3u8 => (), // ok!
    _ => ()
}
```

And finally, for the last example, only `Box<Self>`, `&Self`, `Self`,
or `&mut Self` work as explicit self parameters. Example:

```
struct Foo;

impl Foo {
    fn x(self: Box<Foo>) {} // ok!
}
```
"##,

E0214: r##"
A generic type was described using parentheses rather than angle brackets.
For example:

```compile_fail,E0214
fn main() {
    let v: Vec(&str) = vec!["foo"];
}
```

This is not currently supported: `v` should be defined as `Vec<&str>`.
Parentheses are currently only used with generic types when defining parameters
for `Fn`-family traits.
"##,

E0220: r##"
You used an associated type which isn't defined in the trait.
Erroneous code example:

```compile_fail,E0220
trait T1 {
    type Bar;
}

type Foo = T1<F=i32>; // error: associated type `F` not found for `T1`

// or:

trait T2 {
    type Bar;

    // error: Baz is used but not declared
    fn return_bool(&self, _: &Self::Bar, _: &Self::Baz) -> bool;
}
```

Make sure that you have defined the associated type in the trait body.
Also, verify that you used the right trait or you didn't misspell the
associated type name. Example:

```
trait T1 {
    type Bar;
}

type Foo = T1<Bar=i32>; // ok!

// or:

trait T2 {
    type Bar;
    type Baz; // we declare `Baz` in our trait.

    // and now we can use it here:
    fn return_bool(&self, _: &Self::Bar, _: &Self::Baz) -> bool;
}
```
"##,

E0221: r##"
An attempt was made to retrieve an associated type, but the type was ambiguous.
For example:

```compile_fail,E0221
trait T1 {}
trait T2 {}

trait Foo {
    type A: T1;
}

trait Bar : Foo {
    type A: T2;
    fn do_something() {
        let _: Self::A;
    }
}
```

In this example, `Foo` defines an associated type `A`. `Bar` inherits that type
from `Foo`, and defines another associated type of the same name. As a result,
when we attempt to use `Self::A`, it's ambiguous whether we mean the `A` defined
by `Foo` or the one defined by `Bar`.

There are two options to work around this issue. The first is simply to rename
one of the types. Alternatively, one can specify the intended type using the
following syntax:

```
trait T1 {}
trait T2 {}

trait Foo {
    type A: T1;
}

trait Bar : Foo {
    type A: T2;
    fn do_something() {
        let _: <Self as Bar>::A;
    }
}
```
"##,

E0223: r##"
An attempt was made to retrieve an associated type, but the type was ambiguous.
For example:

```compile_fail,E0223
trait MyTrait {type X; }

fn main() {
    let foo: MyTrait::X;
}
```

The problem here is that we're attempting to take the type of X from MyTrait.
Unfortunately, the type of X is not defined, because it's only made concrete in
implementations of the trait. A working version of this code might look like:

```
trait MyTrait {type X; }
struct MyStruct;

impl MyTrait for MyStruct {
    type X = u32;
}

fn main() {
    let foo: <MyStruct as MyTrait>::X;
}
```

This syntax specifies that we want the X type from MyTrait, as made concrete in
MyStruct. The reason that we cannot simply use `MyStruct::X` is that MyStruct
might implement two different traits with identically-named associated types.
This syntax allows disambiguation between the two.
"##,

E0225: r##"
You attempted to use multiple types as bounds for a closure or trait object.
Rust does not currently support this. A simple example that causes this error:

```compile_fail,E0225
fn main() {
    let _: Box<dyn std::io::Read + std::io::Write>;
}
```

Auto traits such as Send and Sync are an exception to this rule:
It's possible to have bounds of one non-builtin trait, plus any number of
auto traits. For example, the following compiles correctly:

```
fn main() {
    let _: Box<dyn std::io::Read + Send + Sync>;
}
```
"##,

E0229: r##"
An associated type binding was done outside of the type parameter declaration
and `where` clause. Erroneous code example:

```compile_fail,E0229
pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

fn baz<I>(x: &<I as Foo<A=Bar>>::A) {}
// error: associated type bindings are not allowed here
```

To solve this error, please move the type bindings in the type parameter
declaration:

```
# struct Bar;
# trait Foo { type A; }
fn baz<I: Foo<A=Bar>>(x: &<I as Foo>::A) {} // ok!
```

Or in the `where` clause:

```
# struct Bar;
# trait Foo { type A; }
fn baz<I>(x: &<I as Foo>::A) where I: Foo<A=Bar> {}
```
"##,

E0230: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```compile_fail
#![feature(rustc_attrs)]

fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { /* ... */ }

foo(true); // `bool` does not implement `Index<u8>`
```

There will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

As you can see, you can specify type parameters in curly braces for
substitution with the actual types (using the regular format string syntax) in
a given situation. Furthermore, `{Self}` will substitute to the type (in this
case, `bool`) that we tried to use.

This error appears when the curly braces contain an identifier which doesn't
match with any of the type parameters or the string `Self`. This might happen
if you misspelled a type parameter, or if you intended to use literal curly
braces. If it is the latter, escape the curly braces with a second curly brace
of the same type; e.g., a literal `{` is `{{`.
"##,

E0231: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```compile_fail
#![feature(rustc_attrs)]

fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { /* ... */ }

foo(true); // `bool` does not implement `Index<u8>`
```

there will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

As you can see, you can specify type parameters in curly braces for
substitution with the actual types (using the regular format string syntax) in
a given situation. Furthermore, `{Self}` will substitute to the type (in this
case, `bool`) that we tried to use.

This error appears when the curly braces do not contain an identifier. Please
add one of the same name as a type parameter. If you intended to use literal
braces, use `{{` and `}}` to escape them.
"##,

E0232: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```compile_fail
#![feature(rustc_attrs)]

fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { /* ... */ }

foo(true); // `bool` does not implement `Index<u8>`
```

there will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

For this to work, some note must be specified. An empty attribute will not do
anything, please remove the attribute or add some helpful note for users of the
trait.
"##,

E0243: r##"
#### Note: this error code is no longer emitted by the compiler.

This error indicates that not enough type parameters were found in a type or
trait.

For example, the `Foo` struct below is defined to be generic in `T`, but the
type parameter is missing in the definition of `Bar`:

```compile_fail,E0107
struct Foo<T> { x: T }

struct Bar { x: Foo }
```
"##,

E0244: r##"
#### Note: this error code is no longer emitted by the compiler.

This error indicates that too many type parameters were found in a type or
trait.

For example, the `Foo` struct below has no type parameters, but is supplied
with two in the definition of `Bar`:

```compile_fail,E0107
struct Foo { x: bool }

struct Bar<S, T> { x: Foo<S, T> }
```
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

E0261: r##"
When using a lifetime like `'a` in a type, it must be declared before being
used.

These two examples illustrate the problem:

```compile_fail,E0261
// error, use of undeclared lifetime name `'a`
fn foo(x: &'a str) { }

struct Foo {
    // error, use of undeclared lifetime name `'a`
    x: &'a str,
}
```

These can be fixed by declaring lifetime parameters:

```
struct Foo<'a> {
    x: &'a str,
}

fn foo<'a>(x: &'a str) {}
```

Impl blocks declare lifetime parameters separately. You need to add lifetime
parameters to an impl block if you're implementing a type that has a lifetime
parameter of its own.
For example:

```compile_fail,E0261
struct Foo<'a> {
    x: &'a str,
}

// error,  use of undeclared lifetime name `'a`
impl Foo<'a> {
    fn foo<'a>(x: &'a str) {}
}
```

This is fixed by declaring the impl block like this:

```
struct Foo<'a> {
    x: &'a str,
}

// correct
impl<'a> Foo<'a> {
    fn foo(x: &'a str) {}
}
```
"##,

E0262: r##"
Declaring certain lifetime names in parameters is disallowed. For example,
because the `'static` lifetime is a special built-in lifetime name denoting
the lifetime of the entire program, this is an error:

```compile_fail,E0262
// error, invalid lifetime parameter name `'static`
fn foo<'static>(x: &'static str) { }
```
"##,

E0263: r##"
A lifetime name cannot be declared more than once in the same scope. For
example:

```compile_fail,E0263
// error, lifetime name `'a` declared twice in the same scope
fn foo<'a, 'b, 'a>(x: &'a str, y: &'b str) { }
```
"##,

E0264: r##"
An unknown external lang item was used. Erroneous code example:

```compile_fail,E0264
#![feature(lang_items)]

extern "C" {
    #[lang = "cake"] // error: unknown external lang item: `cake`
    fn cake();
}
```

A list of available external lang items is available in
`src/librustc/middle/weak_lang_items.rs`. Example:

```
#![feature(lang_items)]

extern "C" {
    #[lang = "panic_impl"] // ok!
    fn cake();
}
```
"##,

E0267: r##"
This error indicates the use of a loop keyword (`break` or `continue`) inside a
closure but outside of any loop. Erroneous code example:

```compile_fail,E0267
let w = || { break; }; // error: `break` inside of a closure
```

`break` and `continue` keywords can be used as normal inside closures as long as
they are also contained within a loop. To halt the execution of a closure you
should instead use a return statement. Example:

```
let w = || {
    for _ in 0..10 {
        break;
    }
};

w();
```
"##,

E0268: r##"
This error indicates the use of a loop keyword (`break` or `continue`) outside
of a loop. Without a loop to break out of or continue in, no sensible action can
be taken. Erroneous code example:

```compile_fail,E0268
fn some_func() {
    break; // error: `break` outside of a loop
}
```

Please verify that you are using `break` and `continue` only in loops. Example:

```
fn some_func() {
    for _ in 0..10 {
        break; // ok!
    }
}
```
"##,

E0271: r##"
This is because of a type mismatch between the associated type of some
trait (e.g., `T::Bar`, where `T` implements `trait Quux { type Bar; }`)
and another type `U` that is required to be equal to `T::Bar`, but is not.
Examples follow.

Here is a basic example:

```compile_fail,E0271
trait Trait { type AssociatedType; }

fn foo<T>(t: T) where T: Trait<AssociatedType=u32> {
    println!("in foo");
}

impl Trait for i8 { type AssociatedType = &'static str; }

foo(3_i8);
```

Here is that same example again, with some explanatory comments:

```compile_fail,E0271
trait Trait { type AssociatedType; }

fn foo<T>(t: T) where T: Trait<AssociatedType=u32> {
//                    ~~~~~~~~ ~~~~~~~~~~~~~~~~~~
//                        |            |
//         This says `foo` can         |
//           only be used with         |
//              some type that         |
//         implements `Trait`.         |
//                                     |
//                             This says not only must
//                             `T` be an impl of `Trait`
//                             but also that the impl
//                             must assign the type `u32`
//                             to the associated type.
    println!("in foo");
}

impl Trait for i8 { type AssociatedType = &'static str; }
//~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//      |                             |
// `i8` does have                     |
// implementation                     |
// of `Trait`...                      |
//                     ... but it is an implementation
//                     that assigns `&'static str` to
//                     the associated type.

foo(3_i8);
// Here, we invoke `foo` with an `i8`, which does not satisfy
// the constraint `<i8 as Trait>::AssociatedType=u32`, and
// therefore the type-checker complains with this error code.
```

To avoid those issues, you have to make the types match correctly.
So we can fix the previous examples like this:

```
// Basic Example:
trait Trait { type AssociatedType; }

fn foo<T>(t: T) where T: Trait<AssociatedType = &'static str> {
    println!("in foo");
}

impl Trait for i8 { type AssociatedType = &'static str; }

foo(3_i8);

// For-Loop Example:
let vs = vec![1, 2, 3, 4];
for v in &vs {
    match v {
        &1 => {}
        _ => {}
    }
}
```
"##,

E0275: r##"
This error occurs when there was a recursive trait requirement that overflowed
before it could be evaluated. Often this means that there is unbounded
recursion in resolving some type bounds.

For example, in the following code:

```compile_fail,E0275
trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {}
```

To determine if a `T` is `Foo`, we need to check if `Bar<T>` is `Foo`. However,
to do this check, we need to determine that `Bar<Bar<T>>` is `Foo`. To
determine this, we check if `Bar<Bar<Bar<T>>>` is `Foo`, and so on. This is
clearly a recursive requirement that can't be resolved directly.

Consider changing your trait bounds so that they're less self-referential.
"##,

E0276: r##"
This error occurs when a bound in an implementation of a trait does not match
the bounds specified in the original trait. For example:

```compile_fail,E0276
trait Foo {
    fn foo<T>(x: T);
}

impl Foo for bool {
    fn foo<T>(x: T) where T: Copy {}
}
```

Here, all types implementing `Foo` must have a method `foo<T>(x: T)` which can
take any type `T`. However, in the `impl` for `bool`, we have added an extra
bound that `T` is `Copy`, which isn't compatible with the original trait.

Consider removing the bound from the method or adding the bound to the original
method definition in the trait.
"##,

E0277: r##"
You tried to use a type which doesn't implement some trait in a place which
expected that trait. Erroneous code example:

```compile_fail,E0277
// here we declare the Foo trait with a bar method
trait Foo {
    fn bar(&self);
}

// we now declare a function which takes an object implementing the Foo trait
fn some_func<T: Foo>(foo: T) {
    foo.bar();
}

fn main() {
    // we now call the method with the i32 type, which doesn't implement
    // the Foo trait
    some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied
}
```

In order to fix this error, verify that the type you're using does implement
the trait. Example:

```
trait Foo {
    fn bar(&self);
}

fn some_func<T: Foo>(foo: T) {
    foo.bar(); // we can now use this method since i32 implements the
               // Foo trait
}

// we implement the trait on the i32 type
impl Foo for i32 {
    fn bar(&self) {}
}

fn main() {
    some_func(5i32); // ok!
}
```

Or in a generic context, an erroneous code example would look like:

```compile_fail,E0277
fn some_func<T>(foo: T) {
    println!("{:?}", foo); // error: the trait `core::fmt::Debug` is not
                           //        implemented for the type `T`
}

fn main() {
    // We now call the method with the i32 type,
    // which *does* implement the Debug trait.
    some_func(5i32);
}
```

Note that the error here is in the definition of the generic function: Although
we only call it with a parameter that does implement `Debug`, the compiler
still rejects the function: It must work with all possible input types. In
order to make this example compile, we need to restrict the generic type we're
accepting:

```
use std::fmt;

// Restrict the input type to types that implement Debug.
fn some_func<T: fmt::Debug>(foo: T) {
    println!("{:?}", foo);
}

fn main() {
    // Calling the method is still fine, as i32 implements Debug.
    some_func(5i32);

    // This would fail to compile now:
    // struct WithoutDebug;
    // some_func(WithoutDebug);
}
```

Rust only looks at the signature of the called function, as such it must
already specify all requirements that will be used for every type parameter.
"##,

E0281: r##"
#### Note: this error code is no longer emitted by the compiler.

You tried to supply a type which doesn't implement some trait in a location
which expected that trait. This error typically occurs when working with
`Fn`-based types. Erroneous code example:

```compile-fail
fn foo<F: Fn(usize)>(x: F) { }

fn main() {
    // type mismatch: ... implements the trait `core::ops::Fn<(String,)>`,
    // but the trait `core::ops::Fn<(usize,)>` is required
    // [E0281]
    foo(|y: String| { });
}
```

The issue in this case is that `foo` is defined as accepting a `Fn` with one
argument of type `String`, but the closure we attempted to pass to it requires
one arguments of type `usize`.
"##,

E0282: r##"
This error indicates that type inference did not result in one unique possible
type, and extra information is required. In most cases this can be provided
by adding a type annotation. Sometimes you need to specify a generic type
parameter manually.

A common example is the `collect` method on `Iterator`. It has a generic type
parameter with a `FromIterator` bound, which for a `char` iterator is
implemented by `Vec` and `String` among others. Consider the following snippet
that reverses the characters of a string:

```compile_fail,E0282
let x = "hello".chars().rev().collect();
```

In this case, the compiler cannot infer what the type of `x` should be:
`Vec<char>` and `String` are both suitable candidates. To specify which type to
use, you can use a type annotation on `x`:

```
let x: Vec<char> = "hello".chars().rev().collect();
```

It is not necessary to annotate the full type. Once the ambiguity is resolved,
the compiler can infer the rest:

```
let x: Vec<_> = "hello".chars().rev().collect();
```

Another way to provide the compiler with enough information, is to specify the
generic type parameter:

```
let x = "hello".chars().rev().collect::<Vec<char>>();
```

Again, you need not specify the full type if the compiler can infer it:

```
let x = "hello".chars().rev().collect::<Vec<_>>();
```

Apart from a method or function with a generic type parameter, this error can
occur when a type parameter of a struct or trait cannot be inferred. In that
case it is not always possible to use a type annotation, because all candidates
have the same return type. For instance:

```compile_fail,E0282
struct Foo<T> {
    num: T,
}

impl<T> Foo<T> {
    fn bar() -> i32 {
        0
    }

    fn baz() {
        let number = Foo::bar();
    }
}
```

This will fail because the compiler does not know which instance of `Foo` to
call `bar` on. Change `Foo::bar()` to `Foo::<T>::bar()` to resolve the error.
"##,

E0283: r##"
This error occurs when the compiler doesn't have enough information
to unambiguously choose an implementation.

For example:

```compile_fail,E0283
trait Generator {
    fn create() -> u32;
}

struct Impl;

impl Generator for Impl {
    fn create() -> u32 { 1 }
}

struct AnotherImpl;

impl Generator for AnotherImpl {
    fn create() -> u32 { 2 }
}

fn main() {
    let cont: u32 = Generator::create();
    // error, impossible to choose one of Generator trait implementation
    // Should it be Impl or AnotherImpl, maybe something else?
}
```

To resolve this error use the concrete type:

```
trait Generator {
    fn create() -> u32;
}

struct AnotherImpl;

impl Generator for AnotherImpl {
    fn create() -> u32 { 2 }
}

fn main() {
    let gen1 = AnotherImpl::create();

    // if there are multiple methods with same name (different traits)
    let gen2 = <AnotherImpl as Generator>::create();
}
```
"##,

E0284: r##"
This error occurs when the compiler is unable to unambiguously infer the
return type of a function or method which is generic on return type, such
as the `collect` method for `Iterator`s.

For example:

```compile_fail,E0284
fn foo() -> Result<bool, ()> {
    let results = [Ok(true), Ok(false), Err(())].iter().cloned();
    let v: Vec<bool> = results.collect()?;
    // Do things with v...
    Ok(true)
}
```

Here we have an iterator `results` over `Result<bool, ()>`.
Hence, `results.collect()` can return any type implementing
`FromIterator<Result<bool, ()>>`. On the other hand, the
`?` operator can accept any type implementing `Try`.

The author of this code probably wants `collect()` to return a
`Result<Vec<bool>, ()>`, but the compiler can't be sure
that there isn't another type `T` implementing both `Try` and
`FromIterator<Result<bool, ()>>` in scope such that
`T::Ok == Vec<bool>`. Hence, this code is ambiguous and an error
is returned.

To resolve this error, use a concrete type for the intermediate expression:

```
fn foo() -> Result<bool, ()> {
    let results = [Ok(true), Ok(false), Err(())].iter().cloned();
    let v = {
        let temp: Result<Vec<bool>, ()> = results.collect();
        temp?
    };
    // Do things with v...
    Ok(true)
}
```

Note that the type of `v` can now be inferred from the type of `temp`.
"##,

E0297: r##"
#### Note: this error code is no longer emitted by the compiler.

Patterns used to bind names must be irrefutable. That is, they must guarantee
that a name will be extracted in all cases. Instead of pattern matching the
loop variable, consider using a `match` or `if let` inside the loop body. For
instance:

```compile_fail,E0005
let xs : Vec<Option<i32>> = vec![Some(1), None];

// This fails because `None` is not covered.
for Some(x) in xs {
    // ...
}
```

Match inside the loop instead:

```
let xs : Vec<Option<i32>> = vec![Some(1), None];

for item in xs {
    match item {
        Some(x) => {},
        None => {},
    }
}
```

Or use `if let`:

```
let xs : Vec<Option<i32>> = vec![Some(1), None];

for item in xs {
    if let Some(x) = item {
        // ...
    }
}
```
"##,

E0301: r##"
#### Note: this error code is no longer emitted by the compiler.

Mutable borrows are not allowed in pattern guards, because matching cannot have
side effects. Side effects could alter the matched object or the environment
on which the match depends in such a way, that the match would not be
exhaustive. For instance, the following would not match any arm if mutable
borrows were allowed:

```compile_fail,E0596
match Some(()) {
    None => { },
    option if option.take().is_none() => {
        /* impossible, option is `Some` */
    },
    Some(_) => { } // When the previous match failed, the option became `None`.
}
```
"##,

E0302: r##"
#### Note: this error code is no longer emitted by the compiler.

Assignments are not allowed in pattern guards, because matching cannot have
side effects. Side effects could alter the matched object or the environment
on which the match depends in such a way, that the match would not be
exhaustive. For instance, the following would not match any arm if assignments
were allowed:

```compile_fail,E0594
match Some(()) {
    None => { },
    option if { option = None; false } => { },
    Some(_) => { } // When the previous match failed, the option became `None`.
}
```
"##,

E0303: r##"
In certain cases it is possible for sub-bindings to violate memory safety.
Updates to the borrow checker in a future version of Rust may remove this
restriction, but for now patterns must be rewritten without sub-bindings.

Before:

```compile_fail,E0303
match Some("hi".to_string()) {
    ref op_string_ref @ Some(s) => {},
    None => {},
}
```

After:

```
match Some("hi".to_string()) {
    Some(ref s) => {
        let op_string_ref = &Some(s);
        // ...
    },
    None => {},
}
```

The `op_string_ref` binding has type `&Option<&String>` in both cases.

See also https://github.com/rust-lang/rust/issues/14587
"##,

E0307: r##"
This error indicates that the `self` parameter in a method has an invalid
"reciever type".

Methods take a special first parameter, of which there are three variants:
`self`, `&self`, and `&mut self`. These are syntactic sugar for
`self: Self`, `self: &Self`, and `self: &mut Self` respectively.

```
# struct Foo;
trait Trait {
    fn foo(&self);
//         ^^^^^ `self` here is a reference to the receiver object
}

impl Trait for Foo {
    fn foo(&self) {}
//         ^^^^^ the receiver type is `&Foo`
}
```

The type `Self` acts as an alias to the type of the current trait
implementer, or "receiver type". Besides the already mentioned `Self`,
`&Self` and `&mut Self` valid receiver types, the following are also valid:
`self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, and `self: Pin<P>`
(where P is one of the previous types except `Self`). Note that `Self` can
also be the underlying implementing type, like `Foo` in the following
example:

```
# struct Foo;
# trait Trait {
#     fn foo(&self);
# }
impl Trait for Foo {
    fn foo(self: &Foo) {}
}
```

E0307 will be emitted by the compiler when using an invalid reciver type,
like in the following example:

```compile_fail,E0307
# struct Foo;
# struct Bar;
# trait Trait {
#     fn foo(&self);
# }
impl Trait for Foo {
    fn foo(self: &Bar) {}
}
```

The nightly feature [Arbintrary self types][AST] extends the accepted
set of receiver types to also include any type that can dereference to
`Self`:

```
#![feature(arbitrary_self_types)]

struct Foo;
struct Bar;

// Because you can dereference `Bar` into `Foo`...
impl std::ops::Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &Foo
    }
}

impl Foo {
    fn foo(self: Bar) {}
//         ^^^^^^^^^ ...it can be used as the receiver type
}
```

[AST]: https://doc.rust-lang.org/unstable-book/language-features/arbitrary-self-types.html
"##,

E0308: r##"
This error occurs when the compiler was unable to infer the concrete type of a
variable. It can occur for several cases, the most common of which is a
mismatch in the expected type that the compiler inferred for a variable's
initializing expression, and the actual type explicitly assigned to the
variable.

For example:

```compile_fail,E0308
let x: i32 = "I am not a number!";
//     ~~~   ~~~~~~~~~~~~~~~~~~~~
//      |             |
//      |    initializing expression;
//      |    compiler infers type `&str`
//      |
//    type `i32` assigned to variable `x`
```
"##,

E0309: r##"
The type definition contains some field whose type
requires an outlives annotation. Outlives annotations
(e.g., `T: 'a`) are used to guarantee that all the data in T is valid
for at least the lifetime `'a`. This scenario most commonly
arises when the type contains an associated type reference
like `<T as SomeTrait<'a>>::Output`, as shown in this example:

```compile_fail,E0309
// This won't compile because the applicable impl of
// `SomeTrait` (below) requires that `T: 'a`, but the struct does
// not have a matching where-clause.
struct Foo<'a, T> {
    foo: <T as SomeTrait<'a>>::Output,
}

trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T
where
    T: 'a,
{
    type Output = u32;
}
```

Here, the where clause `T: 'a` that appears on the impl is not known to be
satisfied on the struct. To make this example compile, you have to add
a where-clause like `T: 'a` to the struct definition:

```
struct Foo<'a, T>
where
    T: 'a,
{
    foo: <T as SomeTrait<'a>>::Output
}

trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T
where
    T: 'a,
{
    type Output = u32;
}
```
"##,

E0310: r##"
Types in type definitions have lifetimes associated with them that represent
how long the data stored within them is guaranteed to be live. This lifetime
must be as long as the data needs to be alive, and missing the constraint that
denotes this will cause this error.

```compile_fail,E0310
// This won't compile because T is not constrained to the static lifetime
// the reference needs
struct Foo<T> {
    foo: &'static T
}
```

This will compile, because it has the constraint on the type parameter:

```
struct Foo<T: 'static> {
    foo: &'static T
}
```
"##,

E0312: r##"
Reference's lifetime of borrowed content doesn't match the expected lifetime.

Erroneous code example:

```compile_fail,E0312
pub fn opt_str<'a>(maybestr: &'a Option<String>) -> &'static str {
    if maybestr.is_none() {
        "(none)"
    } else {
        let s: &'a str = maybestr.as_ref().unwrap();
        s  // Invalid lifetime!
    }
}
```

To fix this error, either lessen the expected lifetime or find a way to not have
to use this reference outside of its current scope (by running the code directly
in the same block for example?):

```
// In this case, we can fix the issue by switching from "static" lifetime to 'a
pub fn opt_str<'a>(maybestr: &'a Option<String>) -> &'a str {
    if maybestr.is_none() {
        "(none)"
    } else {
        let s: &'a str = maybestr.as_ref().unwrap();
        s  // Ok!
    }
}
```
"##,

E0317: r##"
This error occurs when an `if` expression without an `else` block is used in a
context where a type other than `()` is expected, for example a `let`
expression:

```compile_fail,E0317
fn main() {
    let x = 5;
    let a = if x == 5 { 1 };
}
```

An `if` expression without an `else` block has the type `()`, so this is a type
error. To resolve it, add an `else` block having the same type as the `if`
block.
"##,

E0321: r##"
A cross-crate opt-out trait was implemented on something which wasn't a struct
or enum type. Erroneous code example:

```compile_fail,E0321
#![feature(optin_builtin_traits)]

struct Foo;

impl !Sync for Foo {}

unsafe impl Send for &'static Foo {}
// error: cross-crate traits with a default impl, like `core::marker::Send`,
//        can only be implemented for a struct/enum type, not
//        `&'static Foo`
```

Only structs and enums are permitted to impl Send, Sync, and other opt-out
trait, and the struct or enum must be local to the current crate. So, for
example, `unsafe impl Send for Rc<Foo>` is not allowed.
"##,

E0322: r##"
The `Sized` trait is a special trait built-in to the compiler for types with a
constant size known at compile-time. This trait is automatically implemented
for types as needed by the compiler, and it is currently disallowed to
explicitly implement it for a type.
"##,

E0323: r##"
An associated const was implemented when another trait item was expected.
Erroneous code example:

```compile_fail,E0323
trait Foo {
    type N;
}

struct Bar;

impl Foo for Bar {
    const N : u32 = 0;
    // error: item `N` is an associated const, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

Please verify that the associated const wasn't misspelled and the correct trait
was implemented. Example:

```
struct Bar;

trait Foo {
    type N;
}

impl Foo for Bar {
    type N = u32; // ok!
}
```

Or:

```
struct Bar;

trait Foo {
    const N : u32;
}

impl Foo for Bar {
    const N : u32 = 0; // ok!
}
```
"##,

E0324: r##"
A method was implemented when another trait item was expected. Erroneous
code example:

```compile_fail,E0324
struct Bar;

trait Foo {
    const N : u32;

    fn M();
}

impl Foo for Bar {
    fn N() {}
    // error: item `N` is an associated method, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

To fix this error, please verify that the method name wasn't misspelled and
verify that you are indeed implementing the correct trait items. Example:

```
struct Bar;

trait Foo {
    const N : u32;

    fn M();
}

impl Foo for Bar {
    const N : u32 = 0;

    fn M() {} // ok!
}
```
"##,

E0325: r##"
An associated type was implemented when another trait item was expected.
Erroneous code example:

```compile_fail,E0325
struct Bar;

trait Foo {
    const N : u32;
}

impl Foo for Bar {
    type N = u32;
    // error: item `N` is an associated type, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

Please verify that the associated type name wasn't misspelled and your
implementation corresponds to the trait definition. Example:

```
struct Bar;

trait Foo {
    type N;
}

impl Foo for Bar {
    type N = u32; // ok!
}
```

Or:

```
struct Bar;

trait Foo {
    const N : u32;
}

impl Foo for Bar {
    const N : u32 = 0; // ok!
}
```
"##,

E0326: r##"
The types of any associated constants in a trait implementation must match the
types in the trait definition. This error indicates that there was a mismatch.

Here's an example of this error:

```compile_fail,E0326
trait Foo {
    const BAR: bool;
}

struct Bar;

impl Foo for Bar {
    const BAR: u32 = 5; // error, expected bool, found u32
}
```
"##,

E0328: r##"
The Unsize trait should not be implemented directly. All implementations of
Unsize are provided automatically by the compiler.

Erroneous code example:

```compile_fail,E0328
#![feature(unsize)]

use std::marker::Unsize;

pub struct MyType;

impl<T> Unsize<T> for MyType {}
```

If you are defining your own smart pointer type and would like to enable
conversion from a sized to an unsized type with the
[DST coercion system][RFC 982], use [`CoerceUnsized`] instead.

```
#![feature(coerce_unsized)]

use std::ops::CoerceUnsized;

pub struct MyType<T: ?Sized> {
    field_with_unsized_type: T,
}

impl<T, U> CoerceUnsized<MyType<U>> for MyType<T>
    where T: CoerceUnsized<U> {}
```

[RFC 982]: https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md
[`CoerceUnsized`]: https://doc.rust-lang.org/std/ops/trait.CoerceUnsized.html
"##,

E0329: r##"
#### Note: this error code is no longer emitted by the compiler.

An attempt was made to access an associated constant through either a generic
type parameter or `Self`. This is not supported yet. An example causing this
error is shown below:

```
trait Foo {
    const BAR: f64;
}

struct MyStruct;

impl Foo for MyStruct {
    const BAR: f64 = 0f64;
}

fn get_bar_bad<F: Foo>(t: F) -> f64 {
    F::BAR
}
```

Currently, the value of `BAR` for a particular type can only be accessed
through a concrete type, as shown below:

```
trait Foo {
    const BAR: f64;
}

struct MyStruct;

impl Foo for MyStruct {
    const BAR: f64 = 0f64;
}

fn get_bar_good() -> f64 {
    <MyStruct as Foo>::BAR
}
```
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

E0366: r##"
An attempt was made to implement `Drop` on a concrete specialization of a
generic type. An example is shown below:

```compile_fail,E0366
struct Foo<T> {
    t: T
}

impl Drop for Foo<u32> {
    fn drop(&mut self) {}
}
```

This code is not legal: it is not possible to specialize `Drop` to a subset of
implementations of a generic type. One workaround for this is to wrap the
generic type, as shown below:

```
struct Foo<T> {
    t: T
}

struct Bar {
    t: Foo<u32>
}

impl Drop for Bar {
    fn drop(&mut self) {}
}
```
"##,

E0367: r##"
An attempt was made to implement `Drop` on a specialization of a generic type.
An example is shown below:

```compile_fail,E0367
trait Foo{}

struct MyStruct<T> {
    t: T
}

impl<T: Foo> Drop for MyStruct<T> {
    fn drop(&mut self) {}
}
```

This code is not legal: it is not possible to specialize `Drop` to a subset of
implementations of a generic type. In order for this code to work, `MyStruct`
must also require that `T` implements `Foo`. Alternatively, another option is
to wrap the generic type in another that specializes appropriately:

```
trait Foo{}

struct MyStruct<T> {
    t: T
}

struct MyStructWrapper<T: Foo> {
    t: MyStruct<T>
}

impl <T: Foo> Drop for MyStructWrapper<T> {
    fn drop(&mut self) {}
}
```
"##,

E0368: r##"
This error indicates that a binary assignment operator like `+=` or `^=` was
applied to a type that doesn't support it. For example:

```compile_fail,E0368
let mut x = 12f32; // error: binary operation `<<` cannot be applied to
                   //        type `f32`

x <<= 2;
```

To fix this error, please check that this type implements this binary
operation. Example:

```
let mut x = 12u32; // the `u32` type does implement the `ShlAssign` trait

x <<= 2; // ok!
```

It is also possible to overload most operators for your own type by
implementing the `[OP]Assign` traits from `std::ops`.

Another problem you might be facing is this: suppose you've overloaded the `+`
operator for some type `Foo` by implementing the `std::ops::Add` trait for
`Foo`, but you find that using `+=` does not work, as in this example:

```compile_fail,E0368
use std::ops::Add;

struct Foo(u32);

impl Add for Foo {
    type Output = Foo;

    fn add(self, rhs: Foo) -> Foo {
        Foo(self.0 + rhs.0)
    }
}

fn main() {
    let mut x: Foo = Foo(5);
    x += Foo(7); // error, `+= cannot be applied to the type `Foo`
}
```

This is because `AddAssign` is not automatically implemented, so you need to
manually implement it for your type.
"##,

E0369: r##"
A binary operation was attempted on a type which doesn't support it.
Erroneous code example:

```compile_fail,E0369
let x = 12f32; // error: binary operation `<<` cannot be applied to
               //        type `f32`

x << 2;
```

To fix this error, please check that this type implements this binary
operation. Example:

```
let x = 12u32; // the `u32` type does implement it:
               // https://doc.rust-lang.org/stable/std/ops/trait.Shl.html

x << 2; // ok!
```

It is also possible to overload most operators for your own type by
implementing traits from `std::ops`.

String concatenation appends the string on the right to the string on the
left and may require reallocation. This requires ownership of the string
on the left. If something should be added to a string literal, move the
literal to the heap by allocating it with `to_owned()` like in
`"Your text".to_owned()`.

"##,

E0370: r##"
The maximum value of an enum was reached, so it cannot be automatically
set in the next enum value. Erroneous code example:

```compile_fail,E0370
#[repr(i64)]
enum Foo {
    X = 0x7fffffffffffffff,
    Y, // error: enum discriminant overflowed on value after
       //        9223372036854775807: i64; set explicitly via
       //        Y = -9223372036854775808 if that is desired outcome
}
```

To fix this, please set manually the next enum value or put the enum variant
with the maximum value at the end of the enum. Examples:

```
#[repr(i64)]
enum Foo {
    X = 0x7fffffffffffffff,
    Y = 0, // ok!
}
```

Or:

```
#[repr(i64)]
enum Foo {
    Y = 0, // ok!
    X = 0x7fffffffffffffff,
}
```
"##,

E0371: r##"
When `Trait2` is a subtrait of `Trait1` (for example, when `Trait2` has a
definition like `trait Trait2: Trait1 { ... }`), it is not allowed to implement
`Trait1` for `Trait2`. This is because `Trait2` already implements `Trait1` by
definition, so it is not useful to do this.

Example:

```compile_fail,E0371
trait Foo { fn foo(&self) { } }
trait Bar: Foo { }
trait Baz: Bar { }

impl Bar for Baz { } // error, `Baz` implements `Bar` by definition
impl Foo for Baz { } // error, `Baz` implements `Bar` which implements `Foo`
impl Baz for Baz { } // error, `Baz` (trivially) implements `Baz`
impl Baz for Bar { } // Note: This is OK
```
"##,

E0373: r##"
This error occurs when an attempt is made to use data captured by a closure,
when that data may no longer exist. It's most commonly seen when attempting to
return a closure:

```compile_fail,E0373
fn foo() -> Box<Fn(u32) -> u32> {
    let x = 0u32;
    Box::new(|y| x + y)
}
```

Notice that `x` is stack-allocated by `foo()`. By default, Rust captures
closed-over data by reference. This means that once `foo()` returns, `x` no
longer exists. An attempt to access `x` within the closure would thus be
unsafe.

Another situation where this might be encountered is when spawning threads:

```compile_fail,E0373
fn foo() {
    let x = 0u32;
    let y = 1u32;

    let thr = std::thread::spawn(|| {
        x + y
    });
}
```

Since our new thread runs in parallel, the stack frame containing `x` and `y`
may well have disappeared by the time we try to use them. Even if we call
`thr.join()` within foo (which blocks until `thr` has completed, ensuring the
stack frame won't disappear), we will not succeed: the compiler cannot prove
that this behaviour is safe, and so won't let us do it.

The solution to this problem is usually to switch to using a `move` closure.
This approach moves (or copies, where possible) data into the closure, rather
than taking references to it. For example:

```
fn foo() -> Box<Fn(u32) -> u32> {
    let x = 0u32;
    Box::new(move |y| x + y)
}
```

Now that the closure has its own copy of the data, there's no need to worry
about safety.
"##,

E0374: r##"
A struct without a field containing an unsized type cannot implement
`CoerceUnsized`. An [unsized type][1] is any type that the compiler
doesn't know the length or alignment of at compile time. Any struct
containing an unsized type is also unsized.

[1]: https://doc.rust-lang.org/book/ch19-04-advanced-types.html#dynamically-sized-types-and-the-sized-trait

Example of erroneous code:

```compile_fail,E0374
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: i32,
}

// error: Struct `Foo` has no unsized fields that need `CoerceUnsized`.
impl<T, U> CoerceUnsized<Foo<U>> for Foo<T>
    where T: CoerceUnsized<U> {}
```

`CoerceUnsized` is used to coerce one struct containing an unsized type
into another struct containing a different unsized type. If the struct
doesn't have any fields of unsized types then you don't need explicit
coercion to get the types you want. To fix this you can either
not try to implement `CoerceUnsized` or you can add a field that is
unsized to the struct.

Example:

```
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

// We don't need to impl `CoerceUnsized` here.
struct Foo {
    a: i32,
}

// We add the unsized type field to the struct.
struct Bar<T: ?Sized> {
    a: i32,
    b: T,
}

// The struct has an unsized field so we can implement
// `CoerceUnsized` for it.
impl<T, U> CoerceUnsized<Bar<U>> for Bar<T>
    where T: CoerceUnsized<U> {}
```

Note that `CoerceUnsized` is mainly used by smart pointers like `Box`, `Rc`
and `Arc` to be able to mark that they can coerce unsized types that they
are pointing at.
"##,

E0375: r##"
A struct with more than one field containing an unsized type cannot implement
`CoerceUnsized`. This only occurs when you are trying to coerce one of the
types in your struct to another type in the struct. In this case we try to
impl `CoerceUnsized` from `T` to `U` which are both types that the struct
takes. An [unsized type][1] is any type that the compiler doesn't know the
length or alignment of at compile time. Any struct containing an unsized type
is also unsized.

Example of erroneous code:

```compile_fail,E0375
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized, U: ?Sized> {
    a: i32,
    b: T,
    c: U,
}

// error: Struct `Foo` has more than one unsized field.
impl<T, U> CoerceUnsized<Foo<U, T>> for Foo<T, U> {}
```

`CoerceUnsized` only allows for coercion from a structure with a single
unsized type field to another struct with a single unsized type field.
In fact Rust only allows for a struct to have one unsized type in a struct
and that unsized type must be the last field in the struct. So having two
unsized types in a single struct is not allowed by the compiler. To fix this
use only one field containing an unsized type in the struct and then use
multiple structs to manage each unsized type field you need.

Example:

```
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: i32,
    b: T,
}

impl <T, U> CoerceUnsized<Foo<U>> for Foo<T>
    where T: CoerceUnsized<U> {}

fn coerce_foo<T: CoerceUnsized<U>, U>(t: T) -> Foo<U> {
    Foo { a: 12i32, b: t } // we use coercion to get the `Foo<U>` type we need
}
```

[1]: https://doc.rust-lang.org/book/ch19-04-advanced-types.html#dynamically-sized-types-and-the-sized-trait
"##,

E0376: r##"
The type you are trying to impl `CoerceUnsized` for is not a struct.
`CoerceUnsized` can only be implemented for a struct. Unsized types are
already able to be coerced without an implementation of `CoerceUnsized`
whereas a struct containing an unsized type needs to know the unsized type
field it's containing is able to be coerced. An [unsized type][1]
is any type that the compiler doesn't know the length or alignment of at
compile time. Any struct containing an unsized type is also unsized.

[1]: https://doc.rust-lang.org/book/ch19-04-advanced-types.html#dynamically-sized-types-and-the-sized-trait

Example of erroneous code:

```compile_fail,E0376
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: T,
}

// error: The type `U` is not a struct
impl<T, U> CoerceUnsized<U> for Foo<T> {}
```

The `CoerceUnsized` trait takes a struct type. Make sure the type you are
providing to `CoerceUnsized` is a struct with only the last field containing an
unsized type.

Example:

```
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T> {
    a: T,
}

// The `Foo<U>` is a struct so `CoerceUnsized` can be implemented
impl<T, U> CoerceUnsized<Foo<U>> for Foo<T> where T: CoerceUnsized<U> {}
```

Note that in Rust, structs can only contain an unsized type if the field
containing the unsized type is the last and only unsized type field in the
struct.
"##,

E0378: r##"
The `DispatchFromDyn` trait currently can only be implemented for
builtin pointer types and structs that are newtype wrappers around them
— that is, the struct must have only one field (except for`PhantomData`),
and that field must itself implement `DispatchFromDyn`.

Examples:

```
#![feature(dispatch_from_dyn, unsize)]
use std::{
    marker::Unsize,
    ops::DispatchFromDyn,
};

struct Ptr<T: ?Sized>(*const T);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Ptr<U>> for Ptr<T>
where
    T: Unsize<U>,
{}
```

```
#![feature(dispatch_from_dyn)]
use std::{
    ops::DispatchFromDyn,
    marker::PhantomData,
};

struct Wrapper<T> {
    ptr: T,
    _phantom: PhantomData<()>,
}

impl<T, U> DispatchFromDyn<Wrapper<U>> for Wrapper<T>
where
    T: DispatchFromDyn<U>,
{}
```

Example of illegal `DispatchFromDyn` implementation
(illegal because of extra field)

```compile-fail,E0378
#![feature(dispatch_from_dyn)]
use std::ops::DispatchFromDyn;

struct WrapperExtraField<T> {
    ptr: T,
    extra_stuff: i32,
}

impl<T, U> DispatchFromDyn<WrapperExtraField<U>> for WrapperExtraField<T>
where
    T: DispatchFromDyn<U>,
{}
```
"##,

E0379: r##"
Trait methods cannot be declared `const` by design. For more information, see
[RFC 911].

[RFC 911]: https://github.com/rust-lang/rfcs/pull/911
"##,

E0380: r##"
Auto traits cannot have methods or associated items.
For more information see the [opt-in builtin traits RFC][RFC 19].

[RFC 19]: https://github.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md
"##,

E0381: r##"
It is not allowed to use or capture an uninitialized variable.

Erroneous code example:

```compile_fail,E0381
fn main() {
    let x: i32;
    let y = x; // error, use of possibly-uninitialized variable
}
```

To fix this, ensure that any declared variables are initialized before being
used. Example:

```
fn main() {
    let x: i32 = 0;
    let y = x; // ok!
}
```
"##,

E0382: r##"
This error occurs when an attempt is made to use a variable after its contents
have been moved elsewhere.

Erroneous code example:

```compile_fail,E0382
struct MyStruct { s: u32 }

fn main() {
    let mut x = MyStruct{ s: 5u32 };
    let y = x;
    x.s = 6;
    println!("{}", x.s);
}
```

Since `MyStruct` is a type that is not marked `Copy`, the data gets moved out
of `x` when we set `y`. This is fundamental to Rust's ownership system: outside
of workarounds like `Rc`, a value cannot be owned by more than one variable.

Sometimes we don't need to move the value. Using a reference, we can let another
function borrow the value without changing its ownership. In the example below,
we don't actually have to move our string to `calculate_length`, we can give it
a reference to it with `&` instead.

```
fn main() {
    let s1 = String::from("hello");

    let len = calculate_length(&s1);

    println!("The length of '{}' is {}.", s1, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

A mutable reference can be created with `&mut`.

Sometimes we don't want a reference, but a duplicate. All types marked `Clone`
can be duplicated by calling `.clone()`. Subsequent changes to a clone do not
affect the original variable.

Most types in the standard library are marked `Clone`. The example below
demonstrates using `clone()` on a string. `s1` is first set to "many", and then
copied to `s2`. Then the first character of `s1` is removed, without affecting
`s2`. "any many" is printed to the console.

```
fn main() {
    let mut s1 = String::from("many");
    let s2 = s1.clone();
    s1.remove(0);
    println!("{} {}", s1, s2);
}
```

If we control the definition of a type, we can implement `Clone` on it ourselves
with `#[derive(Clone)]`.

Some types have no ownership semantics at all and are trivial to duplicate. An
example is `i32` and the other number types. We don't have to call `.clone()` to
clone them, because they are marked `Copy` in addition to `Clone`.  Implicit
cloning is more convenient in this case. We can mark our own types `Copy` if
all their members also are marked `Copy`.

In the example below, we implement a `Point` type. Because it only stores two
integers, we opt-out of ownership semantics with `Copy`. Then we can
`let p2 = p1` without `p1` being moved.

```
#[derive(Copy, Clone)]
struct Point { x: i32, y: i32 }

fn main() {
    let mut p1 = Point{ x: -1, y: 2 };
    let p2 = p1;
    p1.x = 1;
    println!("p1: {}, {}", p1.x, p1.y);
    println!("p2: {}, {}", p2.x, p2.y);
}
```

Alternatively, if we don't control the struct's definition, or mutable shared
ownership is truly required, we can use `Rc` and `RefCell`:

```
use std::cell::RefCell;
use std::rc::Rc;

struct MyStruct { s: u32 }

fn main() {
    let mut x = Rc::new(RefCell::new(MyStruct{ s: 5u32 }));
    let y = x.clone();
    x.borrow_mut().s = 6;
    println!("{}", x.borrow().s);
}
```

With this approach, x and y share ownership of the data via the `Rc` (reference
count type). `RefCell` essentially performs runtime borrow checking: ensuring
that at most one writer or multiple readers can access the data at any one time.

If you wish to learn more about ownership in Rust, start with the chapter in the
Book:

https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
"##,

E0383: r##"
#### Note: this error code is no longer emitted by the compiler.

This error occurs when an attempt is made to partially reinitialize a
structure that is currently uninitialized.

For example, this can happen when a drop has taken place:

```compile_fail
struct Foo {
    a: u32,
}
impl Drop for Foo {
    fn drop(&mut self) { /* ... */ }
}

let mut x = Foo { a: 1 };
drop(x); // `x` is now uninitialized
x.a = 2; // error, partial reinitialization of uninitialized structure `t`
```

This error can be fixed by fully reinitializing the structure in question:

```
struct Foo {
    a: u32,
}
impl Drop for Foo {
    fn drop(&mut self) { /* ... */ }
}

let mut x = Foo { a: 1 };
drop(x);
x = Foo { a: 2 };
```
"##,

E0384: r##"
This error occurs when an attempt is made to reassign an immutable variable.

Erroneous code example:

```compile_fail,E0384
fn main() {
    let x = 3;
    x = 5; // error, reassignment of immutable variable
}
```

By default, variables in Rust are immutable. To fix this error, add the keyword
`mut` after the keyword `let` when declaring the variable. For example:

```
fn main() {
    let mut x = 3;
    x = 5;
}
```
"##,

E0386: r##"
#### Note: this error code is no longer emitted by the compiler.

This error occurs when an attempt is made to mutate the target of a mutable
reference stored inside an immutable container.

For example, this can happen when storing a `&mut` inside an immutable `Box`:

```
let mut x: i64 = 1;
let y: Box<_> = Box::new(&mut x);
**y = 2; // error, cannot assign to data in an immutable container
```

This error can be fixed by making the container mutable:

```
let mut x: i64 = 1;
let mut y: Box<_> = Box::new(&mut x);
**y = 2;
```

It can also be fixed by using a type with interior mutability, such as `Cell`
or `RefCell`:

```
use std::cell::Cell;

let x: i64 = 1;
let y: Box<Cell<_>> = Box::new(Cell::new(x));
y.set(2);
```
"##,

E0387: r##"
#### Note: this error code is no longer emitted by the compiler.

This error occurs when an attempt is made to mutate or mutably reference data
that a closure has captured immutably.

Erroneous code example:

```compile_fail
// Accepts a function or a closure that captures its environment immutably.
// Closures passed to foo will not be able to mutate their closed-over state.
fn foo<F: Fn()>(f: F) { }

// Attempts to mutate closed-over data. Error message reads:
// `cannot assign to data in a captured outer variable...`
fn mutable() {
    let mut x = 0u32;
    foo(|| x = 2);
}

// Attempts to take a mutable reference to closed-over data.  Error message
// reads: `cannot borrow data mutably in a captured outer variable...`
fn mut_addr() {
    let mut x = 0u32;
    foo(|| { let y = &mut x; });
}
```

The problem here is that foo is defined as accepting a parameter of type `Fn`.
Closures passed into foo will thus be inferred to be of type `Fn`, meaning that
they capture their context immutably.

If the definition of `foo` is under your control, the simplest solution is to
capture the data mutably. This can be done by defining `foo` to take FnMut
rather than Fn:

```
fn foo<F: FnMut()>(f: F) { }
```

Alternatively, we can consider using the `Cell` and `RefCell` types to achieve
interior mutability through a shared reference. Our example's `mutable`
function could be redefined as below:

```
use std::cell::Cell;

fn foo<F: Fn()>(f: F) { }

fn mutable() {
    let x = Cell::new(0u32);
    foo(|| x.set(2));
}
```

You can read more about cell types in the API documentation:

https://doc.rust-lang.org/std/cell/
"##,

E0388: r##"
#### Note: this error code is no longer emitted by the compiler.
"##,

E0389: r##"
#### Note: this error code is no longer emitted by the compiler.

An attempt was made to mutate data using a non-mutable reference. This
commonly occurs when attempting to assign to a non-mutable reference of a
mutable reference (`&(&mut T)`).

Erroneous code example:

```compile_fail
struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy = FancyNum{ num: 5 };
    let fancy_ref = &(&mut fancy);
    fancy_ref.num = 6; // error: cannot assign to data in a `&` reference
    println!("{}", fancy_ref.num);
}
```

Here, `&mut fancy` is mutable, but `&(&mut fancy)` is not. Creating an
immutable reference to a value borrows it immutably. There can be multiple
references of type `&(&mut T)` that point to the same value, so they must be
immutable to prevent multiple mutable references to the same value.

To fix this, either remove the outer reference:

```
struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy = FancyNum{ num: 5 };

    let fancy_ref = &mut fancy;
    // `fancy_ref` is now &mut FancyNum, rather than &(&mut FancyNum)

    fancy_ref.num = 6; // No error!

    println!("{}", fancy_ref.num);
}
```

Or make the outer reference mutable:

```
struct FancyNum {
    num: u8
}

fn main() {
    let mut fancy = FancyNum{ num: 5 };

    let fancy_ref = &mut (&mut fancy);
    // `fancy_ref` is now &mut(&mut FancyNum), rather than &(&mut FancyNum)

    fancy_ref.num = 6; // No error!

    println!("{}", fancy_ref.num);
}
```
"##,

E0390: r##"
You tried to implement methods for a primitive type. Erroneous code example:

```compile_fail,E0390
struct Foo {
    x: i32
}

impl *mut Foo {}
// error: only a single inherent implementation marked with
//        `#[lang = "mut_ptr"]` is allowed for the `*mut T` primitive
```

This isn't allowed, but using a trait to implement a method is a good solution.
Example:

```
struct Foo {
    x: i32
}

trait Bar {
    fn bar();
}

impl Bar for *mut Foo {
    fn bar() {} // ok!
}
```
"##,

E0391: r##"
This error indicates that some types or traits depend on each other
and therefore cannot be constructed.

The following example contains a circular dependency between two traits:

```compile_fail,E0391
trait FirstTrait : SecondTrait {

}

trait SecondTrait : FirstTrait {

}
```
"##,

E0392: r##"
This error indicates that a type or lifetime parameter has been declared
but not actually used. Here is an example that demonstrates the error:

```compile_fail,E0392
enum Foo<T> {
    Bar,
}
```

If the type parameter was included by mistake, this error can be fixed
by simply removing the type parameter, as shown below:

```
enum Foo {
    Bar,
}
```

Alternatively, if the type parameter was intentionally inserted, it must be
used. A simple fix is shown below:

```
enum Foo<T> {
    Bar(T),
}
```

This error may also commonly be found when working with unsafe code. For
example, when using raw pointers one may wish to specify the lifetime for
which the pointed-at data is valid. An initial attempt (below) causes this
error:

```compile_fail,E0392
struct Foo<'a, T> {
    x: *const T,
}
```

We want to express the constraint that Foo should not outlive `'a`, because
the data pointed to by `T` is only valid for that lifetime. The problem is
that there are no actual uses of `'a`. It's possible to work around this
by adding a PhantomData type to the struct, using it to tell the compiler
to act as if the struct contained a borrowed reference `&'a T`:

```
use std::marker::PhantomData;

struct Foo<'a, T: 'a> {
    x: *const T,
    phantom: PhantomData<&'a T>
}
```

[PhantomData] can also be used to express information about unused type
parameters.

[PhantomData]: https://doc.rust-lang.org/std/marker/struct.PhantomData.html
"##,

E0393: r##"
A type parameter which references `Self` in its default value was not specified.
Example of erroneous code:

```compile_fail,E0393
trait A<T=Self> {}

fn together_we_will_rule_the_galaxy(son: &A) {}
// error: the type parameter `T` must be explicitly specified in an
//        object type because its default value `Self` references the
//        type `Self`
```

A trait object is defined over a single, fully-defined trait. With a regular
default parameter, this parameter can just be substituted in. However, if the
default parameter is `Self`, the trait changes for each concrete type; i.e.
`i32` will be expected to implement `A<i32>`, `bool` will be expected to
implement `A<bool>`, etc... These types will not share an implementation of a
fully-defined trait; instead they share implementations of a trait with
different parameters substituted in for each implementation. This is
irreconcilable with what we need to make a trait object work, and is thus
disallowed. Making the trait concrete by explicitly specifying the value of the
defaulted parameter will fix this issue. Fixed example:

```
trait A<T=Self> {}

fn together_we_will_rule_the_galaxy(son: &A<i32>) {} // Ok!
```
"##,

E0398: r##"
#### Note: this error code is no longer emitted by the compiler.

In Rust 1.3, the default object lifetime bounds are expected to change, as
described in [RFC 1156]. You are getting a warning because the compiler
thinks it is possible that this change will cause a compilation error in your
code. It is possible, though unlikely, that this is a false alarm.

The heart of the change is that where `&'a Box<SomeTrait>` used to default to
`&'a Box<SomeTrait+'a>`, it now defaults to `&'a Box<SomeTrait+'static>` (here,
`SomeTrait` is the name of some trait type). Note that the only types which are
affected are references to boxes, like `&Box<SomeTrait>` or
`&[Box<SomeTrait>]`. More common types like `&SomeTrait` or `Box<SomeTrait>`
are unaffected.

To silence this warning, edit your code to use an explicit bound. Most of the
time, this means that you will want to change the signature of a function that
you are calling. For example, if the error is reported on a call like `foo(x)`,
and `foo` is defined as follows:

```
# trait SomeTrait {}
fn foo(arg: &Box<SomeTrait>) { /* ... */ }
```

You might change it to:

```
# trait SomeTrait {}
fn foo<'a>(arg: &'a Box<SomeTrait+'a>) { /* ... */ }
```

This explicitly states that you expect the trait object `SomeTrait` to contain
references (with a maximum lifetime of `'a`).

[RFC 1156]: https://github.com/rust-lang/rfcs/blob/master/text/1156-adjust-default-object-bounds.md
"##,

E0399: r##"
You implemented a trait, overriding one or more of its associated types but did
not reimplement its default methods.

Example of erroneous code:

```compile_fail,E0399
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
fn f<T, T>(s: T, u: T) {} // error: the name `T` is already used for a generic
                          //        parameter in this item's generic parameters
```

Please verify that none of the type parameters are misspelled, and rename any
clashing parameters. Example:

```
fn f<T, Y>(s: T, u: Y) {} // ok!
```

Type parameters in an associated item also cannot shadow parameters from the
containing item:

```compile_fail,E0403
trait Foo<T> {
    fn do_something(&self) -> T;
    fn do_something_else<T: Clone>(&self, bar: T);
}
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
// error: expected function, tuple struct or tuple variant, found `Foo`
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
// error: expected function, tuple struct or tuple variant,
// found macro `println`
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
The `self` keyword was used inside of an associated function without a "`self`
receiver" parameter.

Erroneous code example:

```compile_fail,E0424
struct Foo;

impl Foo {
    // `bar` is a method, because it has a receiver parameter.
    fn bar(&self) {}

    // `foo` is not a method, because it has no receiver parameter.
    fn foo() {
        self.bar(); // error: `self` value is a keyword only available in
                    //        methods with a `self` parameter
    }
}
```

The `self` keyword can only be used inside methods, which are associated
functions (functions defined inside of a `trait` or `impl` block) that have a
`self` receiver as its first parameter, like `self`, `&self`, `&mut self` or
`self: &mut Pin<Self>` (this last one is an example of an ["abitrary `self`
type"](https://github.com/rust-lang/rust/issues/44874)).

Check if the associated function's parameter list should have contained a `self`
receiver for it to be a method, and add it if so. Example:

```
struct Foo;

impl Foo {
    fn bar(&self) {}

    fn foo(self) { // `foo` is now a method.
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

E0436: r##"
The functional record update syntax is only allowed for structs. (Struct-like
enum variants don't qualify, for example.)

Erroneous code example:

```compile_fail,E0436
enum PublicationFrequency {
    Weekly,
    SemiMonthly { days: (u8, u8), annual_special: bool },
}

fn one_up_competitor(competitor_frequency: PublicationFrequency)
                     -> PublicationFrequency {
    match competitor_frequency {
        PublicationFrequency::Weekly => PublicationFrequency::SemiMonthly {
            days: (1, 15), annual_special: false
        },
        c @ PublicationFrequency::SemiMonthly{ .. } =>
            PublicationFrequency::SemiMonthly {
                annual_special: true, ..c // error: functional record update
                                          //        syntax requires a struct
        }
    }
}
```

Rewrite the expression without functional record update syntax:

```
enum PublicationFrequency {
    Weekly,
    SemiMonthly { days: (u8, u8), annual_special: bool },
}

fn one_up_competitor(competitor_frequency: PublicationFrequency)
                     -> PublicationFrequency {
    match competitor_frequency {
        PublicationFrequency::Weekly => PublicationFrequency::SemiMonthly {
            days: (1, 15), annual_special: false
        },
        PublicationFrequency::SemiMonthly{ days, .. } =>
            PublicationFrequency::SemiMonthly {
                days, annual_special: true // ok!
        }
    }
}
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

E0439: r##"
The length of the platform-intrinsic function `simd_shuffle`
wasn't specified. Erroneous code example:

```compile_fail,E0439
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_shuffle<A,B>(a: A, b: A, c: [u32; 8]) -> B;
    // error: invalid `simd_shuffle`, needs length: `simd_shuffle`
}
```

The `simd_shuffle` function needs the length of the array passed as
last parameter in its name. Example:

```
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_shuffle8<A,B>(a: A, b: A, c: [u32; 8]) -> B;
}
```
"##,

E0445: r##"
A private trait was used on a public type parameter bound.

Erroneous code examples:

```compile_fail,E0445
#![deny(private_in_public)]

trait Foo {
    fn dummy(&self) { }
}

pub trait Bar : Foo {} // error: private trait in public interface
pub struct Bar2<T: Foo>(pub T); // same error
pub fn foo<T: Foo> (t: T) {} // same error
```

To solve this error, please ensure that the trait is also public. The trait
can be made inaccessible if necessary by placing it into a private inner
module, but it still has to be marked with `pub`. Example:

```
pub trait Foo { // we set the Foo trait public
    fn dummy(&self) { }
}

pub trait Bar : Foo {} // ok!
pub struct Bar2<T: Foo>(pub T); // ok!
pub fn foo<T: Foo> (t: T) {} // ok!
```
"##,

E0446: r##"
A private type was used in a public type signature.

Erroneous code example:

```compile_fail,E0446
#![deny(private_in_public)]

mod Foo {
    struct Bar(u32);

    pub fn bar() -> Bar { // error: private type in public interface
        Bar(0)
    }
}
```

To solve this error, please ensure that the type is also public. The type
can be made inaccessible if necessary by placing it into a private inner
module, but it still has to be marked with `pub`.
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
#### Note: this error code is no longer emitted by the compiler.

The `pub` keyword was used inside a function.

Erroneous code example:

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
#### Note: this error code is no longer emitted by the compiler.

The `pub` keyword was used inside a public enum.

Erroneous code example:

```compile_fail
pub enum Foo {
    pub Bar, // error: unnecessary `pub` visibility
}
```

Since the enum is already public, adding `pub` on one its elements is
unnecessary. Example:

```compile_fail
enum Foo {
    pub Bar, // not ok!
}
```

This is the correct syntax:

```
pub enum Foo {
    Bar, // ok!
}
```
"##,

E0449: r##"
A visibility qualifier was used when it was unnecessary. Erroneous code
examples:

```compile_fail,E0449
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
impl Foo for Bar {
    fn foo() {}
}
```
"##,

E0451: r##"
A struct constructor with private fields was invoked.

Erroneous code example:

```compile_fail,E0451
mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize,
    }
}

let f = Bar::Foo{ a: 0, b: 0 }; // error: field `b` of struct `Bar::Foo`
                                //        is private
```

To fix this error, please ensure that all the fields of the struct are public,
or implement a function for easy instantiation. Examples:

```
mod Bar {
    pub struct Foo {
        pub a: isize,
        pub b: isize, // we set `b` field public
    }
}

let f = Bar::Foo{ a: 0, b: 0 }; // ok!
```

Or:

```
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

E0452: r##"
An invalid lint attribute has been given. Erroneous code example:

```compile_fail,E0452
#![allow(foo = "")] // error: malformed lint attribute
```

Lint attributes only accept a list of identifiers (where each identifier is a
lint name). Ensure the attribute is of this form:

```
#![allow(foo)] // ok!
// or:
#![allow(foo, foo2)] // ok!
```
"##,

E0453: r##"
A lint check attribute was overruled by a `forbid` directive set as an
attribute on an enclosing scope, or on the command line with the `-F` option.

Example of erroneous code:

```compile_fail,E0453
#![forbid(non_snake_case)]

#[allow(non_snake_case)]
fn main() {
    let MyNumber = 2; // error: allow(non_snake_case) overruled by outer
                      //        forbid(non_snake_case)
}
```

The `forbid` lint setting, like `deny`, turns the corresponding compiler
warning into a hard error. Unlike `deny`, `forbid` prevents itself from being
overridden by inner attributes.

If you're sure you want to override the lint check, you can change `forbid` to
`deny` (or use `-D` instead of `-F` if the `forbid` setting was given as a
command-line option) to allow the inner lint check attribute:

```
#![deny(non_snake_case)]

#[allow(non_snake_case)]
fn main() {
    let MyNumber = 2; // ok!
}
```

Otherwise, edit the code to pass the lint check, and remove the overruled
attribute:

```
#![forbid(non_snake_case)]

fn main() {
    let my_number = 2;
}
```
"##,

E0454: r##"
A link name was given with an empty name. Erroneous code example:

```ignore (cannot-test-this-because-rustdoc-stops-compile-fail-before-codegen)
#[link(name = "")] extern {}
// error: `#[link(name = "")]` given with empty name
```

The rust compiler cannot link to an external library if you don't give it its
name. Example:

```no_run
#[link(name = "some_lib")] extern {} // ok!
```
"##,

E0455: r##"
Linking with `kind=framework` is only supported when targeting macOS,
as frameworks are specific to that operating system.

Erroneous code example:

```ignore (should-compile_fail-but-cannot-doctest-conditionally-without-macos)
#[link(name = "FooCoreServices", kind = "framework")] extern {}
// OS used to compile is Linux for example
```

To solve this error you can use conditional compilation:

```
#[cfg_attr(target="macos", link(name = "FooCoreServices", kind = "framework"))]
extern {}
```

See more:
https://doc.rust-lang.org/reference/attributes.html#conditional-compilation
"##,

E0458: r##"
An unknown "kind" was specified for a link attribute. Erroneous code example:

```ignore (cannot-test-this-because-rustdoc-stops-compile-fail-before-codegen)
#[link(kind = "wonderful_unicorn")] extern {}
// error: unknown kind: `wonderful_unicorn`
```

Please specify a valid "kind" value, from one of the following:

* static
* dylib
* framework

"##,

E0459: r##"
A link was used without a name parameter. Erroneous code example:

```ignore (cannot-test-this-because-rustdoc-stops-compile-fail-before-codegen)
#[link(kind = "dylib")] extern {}
// error: `#[link(...)]` specified without `name = "foo"`
```

Please add the name parameter to allow the rust compiler to find the library
you want. Example:

```no_run
#[link(kind = "dylib", name = "some_lib")] extern {} // ok!
```
"##,

E0463: r##"
A plugin/crate was declared but cannot be found. Erroneous code example:

```compile_fail,E0463
#![feature(plugin)]
#![plugin(cookie_monster)] // error: can't find crate for `cookie_monster`
extern crate cake_is_a_lie; // error: can't find crate for `cake_is_a_lie`
```

You need to link your code to the relevant crate in order to be able to use it
(through Cargo or the `-L` option of rustc example). Plugins are crates as
well, and you link to them the same way.
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

E0478: r##"
A lifetime bound was not satisfied.

Erroneous code example:

```compile_fail,E0478
// Check that the explicit lifetime bound (`'SnowWhite`, in this example) must
// outlive all the superbounds from the trait (`'kiss`, in this example).

trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite> {
    child: Box<Wedding<'kiss> + 'SnowWhite>,
    // error: lifetime bound not satisfied
}
```

In this example, the `'SnowWhite` lifetime is supposed to outlive the `'kiss`
lifetime but the declaration of the `Prince` struct doesn't enforce it. To fix
this issue, you need to specify it:

```
trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite: 'kiss> { // You say here that 'kiss must live
                                          // longer than 'SnowWhite.
    child: Box<Wedding<'kiss> + 'SnowWhite>, // And now it's all good!
}
```
"##,

E0491: r##"
A reference has a longer lifetime than the data it references.

Erroneous code example:

```compile_fail,E0491
trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T {
    type Output = &'a T; // compile error E0491
}
```

Here, the problem is that a reference type like `&'a T` is only valid
if all the data in T outlives the lifetime `'a`. But this impl as written
is applicable to any lifetime `'a` and any type `T` -- we have no guarantee
that `T` outlives `'a`. To fix this, you can add a where clause like
`where T: 'a`.

```
trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T
where
    T: 'a,
{
    type Output = &'a T; // compile error E0491
}
```
"##,

E0492: r##"
A borrow of a constant containing interior mutability was attempted.

Erroneous code example:

```compile_fail,E0492
use std::sync::atomic::AtomicUsize;

const A: AtomicUsize = AtomicUsize::new(0);
static B: &'static AtomicUsize = &A;
// error: cannot borrow a constant which may contain interior mutability,
//        create a static instead
```

A `const` represents a constant value that should never change. If one takes
a `&` reference to the constant, then one is taking a pointer to some memory
location containing the value. Normally this is perfectly fine: most values
can't be changed via a shared `&` pointer, but interior mutability would allow
it. That is, a constant value could be mutated. On the other hand, a `static` is
explicitly a single memory location, which can be mutated at will.

So, in order to solve this error, either use statics which are `Sync`:

```
use std::sync::atomic::AtomicUsize;

static A: AtomicUsize = AtomicUsize::new(0);
static B: &'static AtomicUsize = &A; // ok!
```

You can also have this error while using a cell type:

```compile_fail,E0492
use std::cell::Cell;

const A: Cell<usize> = Cell::new(1);
const B: &Cell<usize> = &A;
// error: cannot borrow a constant which may contain interior mutability,
//        create a static instead

// or:
struct C { a: Cell<usize> }

const D: C = C { a: Cell::new(1) };
const E: &Cell<usize> = &D.a; // error

// or:
const F: &C = &D; // error
```

This is because cell types do operations that are not thread-safe. Due to this,
they don't implement Sync and thus can't be placed in statics.

However, if you still wish to use these types, you can achieve this by an unsafe
wrapper:

```
use std::cell::Cell;
use std::marker::Sync;

struct NotThreadSafe<T> {
    value: Cell<T>,
}

unsafe impl<T> Sync for NotThreadSafe<T> {}

static A: NotThreadSafe<usize> = NotThreadSafe { value : Cell::new(1) };
static B: &'static NotThreadSafe<usize> = &A; // ok!
```

Remember this solution is unsafe! You will have to ensure that accesses to the
cell are synchronized.
"##,

E0493: r##"
A type with a `Drop` implementation was destructured when trying to initialize
a static item.

Erroneous code example:

```compile_fail,E0493
enum DropType {
    A,
}

impl Drop for DropType {
    fn drop(&mut self) {}
}

struct Foo {
    field1: DropType,
}

static FOO: Foo = Foo { ..Foo { field1: DropType::A } }; // error!
```

The problem here is that if the given type or one of its fields implements the
`Drop` trait, this `Drop` implementation cannot be called during the static
type initialization which might cause a memory leak. To prevent this issue,
you need to instantiate all the static type's fields by hand.

```
enum DropType {
    A,
}

impl Drop for DropType {
    fn drop(&mut self) {}
}

struct Foo {
    field1: DropType,
}

static FOO: Foo = Foo { field1: DropType::A }; // We initialize all fields
                                               // by hand.
```
"##,

E0495: r##"
A lifetime cannot be determined in the given situation.

Erroneous code example:

```compile_fail,E0495
fn transmute_lifetime<'a, 'b, T>(t: &'a (T,)) -> &'b T {
    match (&t,) { // error!
        ((u,),) => u,
    }
}

let y = Box::new((42,));
let x = transmute_lifetime(&y);
```

In this code, you have two ways to solve this issue:
 1. Enforce that `'a` lives at least as long as `'b`.
 2. Use the same lifetime requirement for both input and output values.

So for the first solution, you can do it by replacing `'a` with `'a: 'b`:

```
fn transmute_lifetime<'a: 'b, 'b, T>(t: &'a (T,)) -> &'b T {
    match (&t,) { // ok!
        ((u,),) => u,
    }
}
```

In the second you can do it by simply removing `'b` so they both use `'a`:

```
fn transmute_lifetime<'a, T>(t: &'a (T,)) -> &'a T {
    match (&t,) { // ok!
        ((u,),) => u,
    }
}
```
"##,

E0496: r##"
A lifetime name is shadowing another lifetime name.

Erroneous code example:

```compile_fail,E0496
struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
    fn f<'a>(x: &'a i32) { // error: lifetime name `'a` shadows a lifetime
                           //        name that is already in scope
    }
}
```

Please change the name of one of the lifetimes to remove this error. Example:

```
struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
    fn f<'b>(x: &'b i32) { // ok!
    }
}

fn main() {
}
```
"##,

E0497: r##"
#### Note: this error code is no longer emitted by the compiler.

A stability attribute was used outside of the standard library.

Erroneous code example:

```compile_fail
#[stable] // error: stability attributes may not be used outside of the
          //        standard library
fn foo() {}
```

It is not possible to use stability attributes outside of the standard library.
Also, for now, it is not possible to write deprecation messages either.
"##,

E0499: r##"
A variable was borrowed as mutable more than once.

Erroneous code example:

```compile_fail,E0499
let mut i = 0;
let mut x = &mut i;
let mut a = &mut i;
x;
// error: cannot borrow `i` as mutable more than once at a time
```

Please note that in rust, you can either have many immutable references, or one
mutable reference. Take a look at
https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html for more
information. Example:


```
let mut i = 0;
let mut x = &mut i; // ok!

// or:
let mut i = 0;
let a = &i; // ok!
let b = &i; // still ok!
let c = &i; // still ok!
b;
a;
```
"##,

E0500: r##"
A borrowed variable was used by a closure.

Erroneous code example:

```compile_fail,E0500
fn you_know_nothing(jon_snow: &mut i32) {
    let nights_watch = &jon_snow;
    let starks = || {
        *jon_snow = 3; // error: closure requires unique access to `jon_snow`
                       //        but it is already borrowed
    };
    println!("{}", nights_watch);
}
```

In here, `jon_snow` is already borrowed by the `nights_watch` reference, so it
cannot be borrowed by the `starks` closure at the same time. To fix this issue,
you can create the closure after the borrow has ended:

```
fn you_know_nothing(jon_snow: &mut i32) {
    let nights_watch = &jon_snow;
    println!("{}", nights_watch);
    let starks = || {
        *jon_snow = 3;
    };
}
```

Or, if the type implements the `Clone` trait, you can clone it between
closures:

```
fn you_know_nothing(jon_snow: &mut i32) {
    let mut jon_copy = jon_snow.clone();
    let starks = || {
        *jon_snow = 3;
    };
    println!("{}", jon_copy);
}
```
"##,

E0501: r##"
This error indicates that a mutable variable is being used while it is still
captured by a closure. Because the closure has borrowed the variable, it is not
available for use until the closure goes out of scope.

Note that a capture will either move or borrow a variable, but in this
situation, the closure is borrowing the variable. Take a look at
http://rustbyexample.com/fn/closures/capture.html for more information about
capturing.

Erroneous code example:

```compile_fail,E0501
fn inside_closure(x: &mut i32) {
    // Actions which require unique access
}

fn outside_closure(x: &mut i32) {
    // Actions which require unique access
}

fn foo(a: &mut i32) {
    let mut bar = || {
        inside_closure(a)
    };
    outside_closure(a); // error: cannot borrow `*a` as mutable because previous
                        //        closure requires unique access.
    bar();
}
```

To fix this error, you can finish using the closure before using the captured
variable:

```
fn inside_closure(x: &mut i32) {}
fn outside_closure(x: &mut i32) {}

fn foo(a: &mut i32) {
    let mut bar = || {
        inside_closure(a)
    };
    bar();
    // borrow on `a` ends.
    outside_closure(a); // ok!
}
```

Or you can pass the variable as a parameter to the closure:

```
fn inside_closure(x: &mut i32) {}
fn outside_closure(x: &mut i32) {}

fn foo(a: &mut i32) {
    let mut bar = |s: &mut i32| {
        inside_closure(s)
    };
    outside_closure(a);
    bar(a);
}
```

It may be possible to define the closure later:

```
fn inside_closure(x: &mut i32) {}
fn outside_closure(x: &mut i32) {}

fn foo(a: &mut i32) {
    outside_closure(a);
    let mut bar = || {
        inside_closure(a)
    };
    bar();
}
```
"##,

E0502: r##"
This error indicates that you are trying to borrow a variable as mutable when it
has already been borrowed as immutable.

Erroneous code example:

```compile_fail,E0502
fn bar(x: &mut i32) {}
fn foo(a: &mut i32) {
    let ref y = a; // a is borrowed as immutable.
    bar(a); // error: cannot borrow `*a` as mutable because `a` is also borrowed
            //        as immutable
    println!("{}", y);
}
```

To fix this error, ensure that you don't have any other references to the
variable before trying to access it mutably:

```
fn bar(x: &mut i32) {}
fn foo(a: &mut i32) {
    bar(a);
    let ref y = a; // ok!
    println!("{}", y);
}
```

For more information on the rust ownership system, take a look at
https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html.
"##,

E0503: r##"
A value was used after it was mutably borrowed.

Erroneous code example:

```compile_fail,E0503
fn main() {
    let mut value = 3;
    // Create a mutable borrow of `value`.
    let borrow = &mut value;
    let _sum = value + 1; // error: cannot use `value` because
                          //        it was mutably borrowed
    println!("{}", borrow);
}
```

In this example, `value` is mutably borrowed by `borrow` and cannot be
used to calculate `sum`. This is not possible because this would violate
Rust's mutability rules.

You can fix this error by finishing using the borrow before the next use of
the value:

```
fn main() {
    let mut value = 3;
    let borrow = &mut value;
    println!("{}", borrow);
    // The block has ended and with it the borrow.
    // You can now use `value` again.
    let _sum = value + 1;
}
```

Or by cloning `value` before borrowing it:

```
fn main() {
    let mut value = 3;
    // We clone `value`, creating a copy.
    let value_cloned = value.clone();
    // The mutable borrow is a reference to `value` and
    // not to `value_cloned`...
    let borrow = &mut value;
    // ... which means we can still use `value_cloned`,
    let _sum = value_cloned + 1;
    // even though the borrow only ends here.
    println!("{}", borrow);
}
```

You can find more information about borrowing in the rust-book:
http://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html
"##,

E0504: r##"
#### Note: this error code is no longer emitted by the compiler.

This error occurs when an attempt is made to move a borrowed variable into a
closure.

Erroneous code example:

```compile_fail
struct FancyNum {
    num: u8,
}

fn main() {
    let fancy_num = FancyNum { num: 5 };
    let fancy_ref = &fancy_num;

    let x = move || {
        println!("child function: {}", fancy_num.num);
        // error: cannot move `fancy_num` into closure because it is borrowed
    };

    x();
    println!("main function: {}", fancy_ref.num);
}
```

Here, `fancy_num` is borrowed by `fancy_ref` and so cannot be moved into
the closure `x`. There is no way to move a value into a closure while it is
borrowed, as that would invalidate the borrow.

If the closure can't outlive the value being moved, try using a reference
rather than moving:

```
struct FancyNum {
    num: u8,
}

fn main() {
    let fancy_num = FancyNum { num: 5 };
    let fancy_ref = &fancy_num;

    let x = move || {
        // fancy_ref is usable here because it doesn't move `fancy_num`
        println!("child function: {}", fancy_ref.num);
    };

    x();

    println!("main function: {}", fancy_num.num);
}
```

If the value has to be borrowed and then moved, try limiting the lifetime of
the borrow using a scoped block:

```
struct FancyNum {
    num: u8,
}

fn main() {
    let fancy_num = FancyNum { num: 5 };

    {
        let fancy_ref = &fancy_num;
        println!("main function: {}", fancy_ref.num);
        // `fancy_ref` goes out of scope here
    }

    let x = move || {
        // `fancy_num` can be moved now (no more references exist)
        println!("child function: {}", fancy_num.num);
    };

    x();
}
```

If the lifetime of a reference isn't enough, such as in the case of threading,
consider using an `Arc` to create a reference-counted value:

```
use std::sync::Arc;
use std::thread;

struct FancyNum {
    num: u8,
}

fn main() {
    let fancy_ref1 = Arc::new(FancyNum { num: 5 });
    let fancy_ref2 = fancy_ref1.clone();

    let x = thread::spawn(move || {
        // `fancy_ref1` can be moved and has a `'static` lifetime
        println!("child thread: {}", fancy_ref1.num);
    });

    x.join().expect("child thread should finish");
    println!("main thread: {}", fancy_ref2.num);
}
```
"##,

E0505: r##"
A value was moved out while it was still borrowed.

Erroneous code example:

```compile_fail,E0505
struct Value {}

fn borrow(val: &Value) {}

fn eat(val: Value) {}

fn main() {
    let x = Value{};
    let _ref_to_val: &Value = &x;
    eat(x);
    borrow(_ref_to_val);
}
```

Here, the function `eat` takes ownership of `x`. However,
`x` cannot be moved because the borrow to `_ref_to_val`
needs to last till the function `borrow`.
To fix that you can do a few different things:

* Try to avoid moving the variable.
* Release borrow before move.
* Implement the `Copy` trait on the type.

Examples:

```
struct Value {}

fn borrow(val: &Value) {}

fn eat(val: &Value) {}

fn main() {
    let x = Value{};

    let ref_to_val: &Value = &x;
    eat(&x); // pass by reference, if it's possible
    borrow(ref_to_val);
}
```

Or:

```
struct Value {}

fn borrow(val: &Value) {}

fn eat(val: Value) {}

fn main() {
    let x = Value{};

    let ref_to_val: &Value = &x;
    borrow(ref_to_val);
    // ref_to_val is no longer used.
    eat(x);
}
```

Or:

```
#[derive(Clone, Copy)] // implement Copy trait
struct Value {}

fn borrow(val: &Value) {}

fn eat(val: Value) {}

fn main() {
    let x = Value{};
    let ref_to_val: &Value = &x;
    eat(x); // it will be copied here.
    borrow(ref_to_val);
}
```

You can find more information about borrowing in the rust-book:
http://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html
"##,

E0506: r##"
This error occurs when an attempt is made to assign to a borrowed value.

Erroneous code example:

```compile_fail,E0506
struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy_num = FancyNum { num: 5 };
    let fancy_ref = &fancy_num;
    fancy_num = FancyNum { num: 6 };
    // error: cannot assign to `fancy_num` because it is borrowed

    println!("Num: {}, Ref: {}", fancy_num.num, fancy_ref.num);
}
```

Because `fancy_ref` still holds a reference to `fancy_num`, `fancy_num` can't
be assigned to a new value as it would invalidate the reference.

Alternatively, we can move out of `fancy_num` into a second `fancy_num`:

```
struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy_num = FancyNum { num: 5 };
    let moved_num = fancy_num;
    fancy_num = FancyNum { num: 6 };

    println!("Num: {}, Moved num: {}", fancy_num.num, moved_num.num);
}
```

If the value has to be borrowed, try limiting the lifetime of the borrow using
a scoped block:

```
struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy_num = FancyNum { num: 5 };

    {
        let fancy_ref = &fancy_num;
        println!("Ref: {}", fancy_ref.num);
    }

    // Works because `fancy_ref` is no longer in scope
    fancy_num = FancyNum { num: 6 };
    println!("Num: {}", fancy_num.num);
}
```

Or by moving the reference into a function:

```
struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy_num = FancyNum { num: 5 };

    print_fancy_ref(&fancy_num);

    // Works because function borrow has ended
    fancy_num = FancyNum { num: 6 };
    println!("Num: {}", fancy_num.num);
}

fn print_fancy_ref(fancy_ref: &FancyNum){
    println!("Ref: {}", fancy_ref.num);
}
```
"##,

E0507: r##"
You tried to move out of a value which was borrowed.

This can also happen when using a type implementing `Fn` or `FnMut`, as neither
allows moving out of them (they usually represent closures which can be called
more than once). Much of the text following applies equally well to non-`FnOnce`
closure bodies.

Erroneous code example:

```compile_fail,E0507
use std::cell::RefCell;

struct TheDarkKnight;

impl TheDarkKnight {
    fn nothing_is_true(self) {}
}

fn main() {
    let x = RefCell::new(TheDarkKnight);

    x.borrow().nothing_is_true(); // error: cannot move out of borrowed content
}
```

Here, the `nothing_is_true` method takes the ownership of `self`. However,
`self` cannot be moved because `.borrow()` only provides an `&TheDarkKnight`,
which is a borrow of the content owned by the `RefCell`. To fix this error,
you have three choices:

* Try to avoid moving the variable.
* Somehow reclaim the ownership.
* Implement the `Copy` trait on the type.

Examples:

```
use std::cell::RefCell;

struct TheDarkKnight;

impl TheDarkKnight {
    fn nothing_is_true(&self) {} // First case, we don't take ownership
}

fn main() {
    let x = RefCell::new(TheDarkKnight);

    x.borrow().nothing_is_true(); // ok!
}
```

Or:

```
use std::cell::RefCell;

struct TheDarkKnight;

impl TheDarkKnight {
    fn nothing_is_true(self) {}
}

fn main() {
    let x = RefCell::new(TheDarkKnight);
    let x = x.into_inner(); // we get back ownership

    x.nothing_is_true(); // ok!
}
```

Or:

```
use std::cell::RefCell;

#[derive(Clone, Copy)] // we implement the Copy trait
struct TheDarkKnight;

impl TheDarkKnight {
    fn nothing_is_true(self) {}
}

fn main() {
    let x = RefCell::new(TheDarkKnight);

    x.borrow().nothing_is_true(); // ok!
}
```

Moving a member out of a mutably borrowed struct will also cause E0507 error:

```compile_fail,E0507
struct TheDarkKnight;

impl TheDarkKnight {
    fn nothing_is_true(self) {}
}

struct Batcave {
    knight: TheDarkKnight
}

fn main() {
    let mut cave = Batcave {
        knight: TheDarkKnight
    };
    let borrowed = &mut cave;

    borrowed.knight.nothing_is_true(); // E0507
}
```

It is fine only if you put something back. `mem::replace` can be used for that:

```
# struct TheDarkKnight;
# impl TheDarkKnight { fn nothing_is_true(self) {} }
# struct Batcave { knight: TheDarkKnight }
use std::mem;

let mut cave = Batcave {
    knight: TheDarkKnight
};
let borrowed = &mut cave;

mem::replace(&mut borrowed.knight, TheDarkKnight).nothing_is_true(); // ok!
```

You can find more information about borrowing in the rust-book:
http://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html
"##,

E0508: r##"
A value was moved out of a non-copy fixed-size array.

Erroneous code example:

```compile_fail,E0508
struct NonCopy;

fn main() {
    let array = [NonCopy; 1];
    let _value = array[0]; // error: cannot move out of type `[NonCopy; 1]`,
                           //        a non-copy fixed-size array
}
```

The first element was moved out of the array, but this is not
possible because `NonCopy` does not implement the `Copy` trait.

Consider borrowing the element instead of moving it:

```
struct NonCopy;

fn main() {
    let array = [NonCopy; 1];
    let _value = &array[0]; // Borrowing is allowed, unlike moving.
}
```

Alternatively, if your type implements `Clone` and you need to own the value,
consider borrowing and then cloning:

```
#[derive(Clone)]
struct NonCopy;

fn main() {
    let array = [NonCopy; 1];
    // Now you can clone the array element.
    let _value = array[0].clone();
}
```
"##,

E0509: r##"
This error occurs when an attempt is made to move out of a value whose type
implements the `Drop` trait.

Erroneous code example:

```compile_fail,E0509
struct FancyNum {
    num: usize
}

struct DropStruct {
    fancy: FancyNum
}

impl Drop for DropStruct {
    fn drop(&mut self) {
        // Destruct DropStruct, possibly using FancyNum
    }
}

fn main() {
    let drop_struct = DropStruct{fancy: FancyNum{num: 5}};
    let fancy_field = drop_struct.fancy; // Error E0509
    println!("Fancy: {}", fancy_field.num);
    // implicit call to `drop_struct.drop()` as drop_struct goes out of scope
}
```

Here, we tried to move a field out of a struct of type `DropStruct` which
implements the `Drop` trait. However, a struct cannot be dropped if one or
more of its fields have been moved.

Structs implementing the `Drop` trait have an implicit destructor that gets
called when they go out of scope. This destructor may use the fields of the
struct, so moving out of the struct could make it impossible to run the
destructor. Therefore, we must think of all values whose type implements the
`Drop` trait as single units whose fields cannot be moved.

This error can be fixed by creating a reference to the fields of a struct,
enum, or tuple using the `ref` keyword:

```
struct FancyNum {
    num: usize
}

struct DropStruct {
    fancy: FancyNum
}

impl Drop for DropStruct {
    fn drop(&mut self) {
        // Destruct DropStruct, possibly using FancyNum
    }
}

fn main() {
    let drop_struct = DropStruct{fancy: FancyNum{num: 5}};
    let ref fancy_field = drop_struct.fancy; // No more errors!
    println!("Fancy: {}", fancy_field.num);
    // implicit call to `drop_struct.drop()` as drop_struct goes out of scope
}
```

Note that this technique can also be used in the arms of a match expression:

```
struct FancyNum {
    num: usize
}

enum DropEnum {
    Fancy(FancyNum)
}

impl Drop for DropEnum {
    fn drop(&mut self) {
        // Destruct DropEnum, possibly using FancyNum
    }
}

fn main() {
    // Creates and enum of type `DropEnum`, which implements `Drop`
    let drop_enum = DropEnum::Fancy(FancyNum{num: 10});
    match drop_enum {
        // Creates a reference to the inside of `DropEnum::Fancy`
        DropEnum::Fancy(ref fancy_field) => // No error!
            println!("It was fancy-- {}!", fancy_field.num),
    }
    // implicit call to `drop_enum.drop()` as drop_enum goes out of scope
}
```
"##,

E0510: r##"
Cannot mutate place in this match guard.

When matching on a variable it cannot be mutated in the match guards, as this
could cause the match to be non-exhaustive:

```compile_fail,E0510
let mut x = Some(0);
match x {
    None => (),
    Some(_) if { x = None; false } => (),
    Some(v) => (), // No longer matches
}
```

Here executing `x = None` would modify the value being matched and require us
to go "back in time" to the `None` arm.
"##,

E0511: r##"
Invalid monomorphization of an intrinsic function was used. Erroneous code
example:

```ignore (error-emitted-at-codegen-which-cannot-be-handled-by-compile_fail)
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

fn main() {
    unsafe { simd_add(0, 1); }
    // error: invalid monomorphization of `simd_add` intrinsic
}
```

The generic type has to be a SIMD type. Example:

```
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x2(i32, i32);

extern "platform-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

unsafe { simd_add(i32x2(0, 0), i32x2(1, 2)); } // ok!
```
"##,

E0512: r##"
Transmute with two differently sized types was attempted. Erroneous code
example:

```compile_fail,E0512
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0u16)); }
    // error: cannot transmute between types of different sizes,
    //        or dependently-sized types
}
```

Please use types with same size or use the expected type directly. Example:

```
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0i8)); } // ok!
    // or:
    unsafe { takes_u8(0u8); } // ok!
}
```
"##,

E0515: r##"
Cannot return value that references local variable

Local variables, function parameters and temporaries are all dropped before the
end of the function body. So a reference to them cannot be returned.

Erroneous code example:

```compile_fail,E0515
fn get_dangling_reference() -> &'static i32 {
    let x = 0;
    &x
}
```

```compile_fail,E0515
use std::slice::Iter;
fn get_dangling_iterator<'a>() -> Iter<'a, i32> {
    let v = vec![1, 2, 3];
    v.iter()
}
```

Consider returning an owned value instead:

```
use std::vec::IntoIter;

fn get_integer() -> i32 {
    let x = 0;
    x
}

fn get_owned_iterator() -> IntoIter<i32> {
    let v = vec![1, 2, 3];
    v.into_iter()
}
```
"##,

E0516: r##"
The `typeof` keyword is currently reserved but unimplemented.
Erroneous code example:

```compile_fail,E0516
fn main() {
    let x: typeof(92) = 92;
}
```

Try using type inference instead. Example:

```
fn main() {
    let x = 92;
}
```
"##,

E0517: r##"
This error indicates that a `#[repr(..)]` attribute was placed on an
unsupported item.

Examples of erroneous code:

```compile_fail,E0517
#[repr(C)]
type Foo = u8;

#[repr(packed)]
enum Foo {Bar, Baz}

#[repr(u8)]
struct Foo {bar: bool, baz: bool}

#[repr(C)]
impl Foo {
    // ...
}
```

* The `#[repr(C)]` attribute can only be placed on structs and enums.
* The `#[repr(packed)]` and `#[repr(simd)]` attributes only work on structs.
* The `#[repr(u8)]`, `#[repr(i16)]`, etc attributes only work on enums.

These attributes do not work on typedefs, since typedefs are just aliases.

Representations like `#[repr(u8)]`, `#[repr(i64)]` are for selecting the
discriminant size for enums with no data fields on any of the variants, e.g.
`enum Color {Red, Blue, Green}`, effectively setting the size of the enum to
the size of the provided type. Such an enum can be cast to a value of the same
type as well. In short, `#[repr(u8)]` makes the enum behave like an integer
with a constrained set of allowed values.

Only field-less enums can be cast to numerical primitives, so this attribute
will not apply to structs.

`#[repr(packed)]` reduces padding to make the struct size smaller. The
representation of enums isn't strictly defined in Rust, and this attribute
won't work on enums.

`#[repr(simd)]` will give a struct consisting of a homogeneous series of machine
types (i.e., `u8`, `i32`, etc) a representation that permits vectorization via
SIMD. This doesn't make much sense for enums since they don't consist of a
single list of data.
"##,

E0518: r##"
This error indicates that an `#[inline(..)]` attribute was incorrectly placed
on something other than a function or method.

Examples of erroneous code:

```compile_fail,E0518
#[inline(always)]
struct Foo;

#[inline(never)]
impl Foo {
    // ...
}
```

`#[inline]` hints the compiler whether or not to attempt to inline a method or
function. By default, the compiler does a pretty good job of figuring this out
itself, but if you feel the need for annotations, `#[inline(always)]` and
`#[inline(never)]` can override or force the compiler's decision.

If you wish to apply this attribute to all methods in an impl, manually annotate
each method; it is not possible to annotate the entire impl with an `#[inline]`
attribute.
"##,

E0520: r##"
A non-default implementation was already made on this type so it cannot be
specialized further. Erroneous code example:

```compile_fail,E0520
#![feature(specialization)]

trait SpaceLlama {
    fn fly(&self);
}

// applies to all T
impl<T> SpaceLlama for T {
    default fn fly(&self) {}
}

// non-default impl
// applies to all `Clone` T and overrides the previous impl
impl<T: Clone> SpaceLlama for T {
    fn fly(&self) {}
}

// since `i32` is clone, this conflicts with the previous implementation
impl SpaceLlama for i32 {
    default fn fly(&self) {}
    // error: item `fly` is provided by an `impl` that specializes
    //        another, but the item in the parent `impl` is not marked
    //        `default` and so it cannot be specialized.
}
```

Specialization only allows you to override `default` functions in
implementations.

To fix this error, you need to mark all the parent implementations as default.
Example:

```
#![feature(specialization)]

trait SpaceLlama {
    fn fly(&self);
}

// applies to all T
impl<T> SpaceLlama for T {
    default fn fly(&self) {} // This is a parent implementation.
}

// applies to all `Clone` T; overrides the previous impl
impl<T: Clone> SpaceLlama for T {
    default fn fly(&self) {} // This is a parent implementation but was
                             // previously not a default one, causing the error
}

// applies to i32, overrides the previous two impls
impl SpaceLlama for i32 {
    fn fly(&self) {} // And now that's ok!
}
```
"##,

E0522: r##"
The lang attribute is intended for marking special items that are built-in to
Rust itself. This includes special traits (like `Copy` and `Sized`) that affect
how the compiler behaves, as well as special functions that may be automatically
invoked (such as the handler for out-of-bounds accesses when indexing a slice).
Erroneous code example:

```compile_fail,E0522
#![feature(lang_items)]

#[lang = "cookie"]
fn cookie() -> ! { // error: definition of an unknown language item: `cookie`
    loop {}
}
```
"##,

E0524: r##"
A variable which requires unique access is being used in more than one closure
at the same time.

Erroneous code example:

```compile_fail,E0524
fn set(x: &mut isize) {
    *x += 4;
}

fn dragoooon(x: &mut isize) {
    let mut c1 = || set(x);
    let mut c2 = || set(x); // error!

    c2();
    c1();
}
```

To solve this issue, multiple solutions are available. First, is it required
for this variable to be used in more than one closure at a time? If it is the
case, use reference counted types such as `Rc` (or `Arc` if it runs
concurrently):

```
use std::rc::Rc;
use std::cell::RefCell;

fn set(x: &mut isize) {
    *x += 4;
}

fn dragoooon(x: &mut isize) {
    let x = Rc::new(RefCell::new(x));
    let y = Rc::clone(&x);
    let mut c1 = || { let mut x2 = x.borrow_mut(); set(&mut x2); };
    let mut c2 = || { let mut x2 = y.borrow_mut(); set(&mut x2); }; // ok!

    c2();
    c1();
}
```

If not, just run closures one at a time:

```
fn set(x: &mut isize) {
    *x += 4;
}

fn dragoooon(x: &mut isize) {
    { // This block isn't necessary since non-lexical lifetimes, it's just to
      // make it more clear.
        let mut c1 = || set(&mut *x);
        c1();
    } // `c1` has been dropped here so we're free to use `x` again!
    let mut c2 = || set(&mut *x);
    c2();
}
```
"##,

E0525: r##"
A closure was used but didn't implement the expected trait.

Erroneous code example:

```compile_fail,E0525
struct X;

fn foo<T>(_: T) {}
fn bar<T: Fn(u32)>(_: T) {}

fn main() {
    let x = X;
    let closure = |_| foo(x); // error: expected a closure that implements
                              //        the `Fn` trait, but this closure only
                              //        implements `FnOnce`
    bar(closure);
}
```

In the example above, `closure` is an `FnOnce` closure whereas the `bar`
function expected an `Fn` closure. In this case, it's simple to fix the issue,
you just have to implement `Copy` and `Clone` traits on `struct X` and it'll
be ok:

```
#[derive(Clone, Copy)] // We implement `Clone` and `Copy` traits.
struct X;

fn foo<T>(_: T) {}
fn bar<T: Fn(u32)>(_: T) {}

fn main() {
    let x = X;
    let closure = |_| foo(x);
    bar(closure); // ok!
}
```

To understand better how closures work in Rust, read:
https://doc.rust-lang.org/book/ch13-01-closures.html
"##,

E0527: r##"
The number of elements in an array or slice pattern differed from the number of
elements in the array being matched.

Example of erroneous code:

```compile_fail,E0527
let r = &[1, 2, 3, 4];
match r {
    &[a, b] => { // error: pattern requires 2 elements but array
                 //        has 4
        println!("a={}, b={}", a, b);
    }
}
```

Ensure that the pattern is consistent with the size of the matched
array. Additional elements can be matched with `..`:

```
#![feature(slice_patterns)]

let r = &[1, 2, 3, 4];
match r {
    &[a, b, ..] => { // ok!
        println!("a={}, b={}", a, b);
    }
}
```
"##,

E0528: r##"
An array or slice pattern required more elements than were present in the
matched array.

Example of erroneous code:

```compile_fail,E0528
#![feature(slice_patterns)]

let r = &[1, 2];
match r {
    &[a, b, c, rest @ ..] => { // error: pattern requires at least 3
                               //        elements but array has 2
        println!("a={}, b={}, c={} rest={:?}", a, b, c, rest);
    }
}
```

Ensure that the matched array has at least as many elements as the pattern
requires. You can match an arbitrary number of remaining elements with `..`:

```
#![feature(slice_patterns)]

let r = &[1, 2, 3, 4, 5];
match r {
    &[a, b, c, rest @ ..] => { // ok!
        // prints `a=1, b=2, c=3 rest=[4, 5]`
        println!("a={}, b={}, c={} rest={:?}", a, b, c, rest);
    }
}
```
"##,

E0529: r##"
An array or slice pattern was matched against some other type.

Example of erroneous code:

```compile_fail,E0529
let r: f32 = 1.0;
match r {
    [a, b] => { // error: expected an array or slice, found `f32`
        println!("a={}, b={}", a, b);
    }
}
```

Ensure that the pattern and the expression being matched on are of consistent
types:

```
let r = [1.0, 2.0];
match r {
    [a, b] => { // ok!
        println!("a={}, b={}", a, b);
    }
}
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

E0531: r##"
An unknown tuple struct/variant has been used.

Erroneous code example:

```compile_fail,E0531
let Type(x) = Type(12); // error!
match Bar(12) {
    Bar(x) => {} // error!
    _ => {}
}
```

In most cases, it's either a forgotten import or a typo. However, let's look at
how you can have such a type:

```edition2018
struct Type(u32); // this is a tuple struct

enum Foo {
    Bar(u32), // this is a tuple variant
}

use Foo::*; // To use Foo's variant directly, we need to import them in
            // the scope.
```

Either way, it should work fine with our previous code:

```edition2018
struct Type(u32);

enum Foo {
    Bar(u32),
}
use Foo::*;

let Type(x) = Type(12); // ok!
match Type(12) {
    Type(x) => {} // ok!
    _ => {}
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
        // error: expected unit struct, unit variant or constant, found tuple
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

E0533: r##"
An item which isn't a unit struct, a variant, nor a constant has been used as a
match pattern.

Erroneous code example:

```compile_fail,E0533
struct Tortoise;

impl Tortoise {
    fn turtle(&self) -> u32 { 0 }
}

match 0u32 {
    Tortoise::turtle => {} // Error!
    _ => {}
}
if let Tortoise::turtle = 0u32 {} // Same error!
```

If you want to match against a value returned by a method, you need to bind the
value first:

```
struct Tortoise;

impl Tortoise {
    fn turtle(&self) -> u32 { 0 }
}

match 0u32 {
    x if x == Tortoise.turtle() => {} // Bound into `x` then we compare it!
    _ => {}
}
```
"##,

E0534: r##"
The `inline` attribute was malformed.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
#[inline()] // error: expected one argument
pub fn something() {}

fn main() {}
```

The parenthesized `inline` attribute requires the parameter to be specified:

```
#[inline(always)]
fn something() {}
```

or:

```
#[inline(never)]
fn something() {}
```

Alternatively, a paren-less version of the attribute may be used to hint the
compiler about inlining opportunity:

```
#[inline]
fn something() {}
```

For more information about the inline attribute, read:
https://doc.rust-lang.org/reference.html#inline-attributes
"##,

E0535: r##"
An unknown argument was given to the `inline` attribute.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
#[inline(unknown)] // error: invalid argument
pub fn something() {}

fn main() {}
```

The `inline` attribute only supports two arguments:

 * always
 * never

All other arguments given to the `inline` attribute will return this error.
Example:

```
#[inline(never)] // ok!
pub fn something() {}

fn main() {}
```

For more information about the inline attribute, https:
read://doc.rust-lang.org/reference.html#inline-attributes
"##,

E0536: r##"
The `not` cfg-predicate was malformed.

Erroneous code example:

```compile_fail,E0536
#[cfg(not())] // error: expected 1 cfg-pattern
pub fn something() {}

pub fn main() {}
```

The `not` predicate expects one cfg-pattern. Example:

```
#[cfg(not(target_os = "linux"))] // ok!
pub fn something() {}

pub fn main() {}
```

For more information about the cfg attribute, read:
https://doc.rust-lang.org/reference.html#conditional-compilation
"##,

E0537: r##"
An unknown predicate was used inside the `cfg` attribute.

Erroneous code example:

```compile_fail,E0537
#[cfg(unknown())] // error: invalid predicate `unknown`
pub fn something() {}

pub fn main() {}
```

The `cfg` attribute supports only three kinds of predicates:

 * any
 * all
 * not

Example:

```
#[cfg(not(target_os = "linux"))] // ok!
pub fn something() {}

pub fn main() {}
```

For more information about the cfg attribute, read:
https://doc.rust-lang.org/reference.html#conditional-compilation
"##,

E0538: r##"
Attribute contains same meta item more than once.

Erroneous code example:

```compile_fail,E0538
#[deprecated(
    since="1.0.0",
    note="First deprecation note.",
    note="Second deprecation note." // error: multiple same meta item
)]
fn deprecated_function() {}
```

Meta items are the key-value pairs inside of an attribute. Each key may only be
used once in each attribute.

To fix the problem, remove all but one of the meta items with the same key.

Example:

```
#[deprecated(
    since="1.0.0",
    note="First deprecation note."
)]
fn deprecated_function() {}
```
"##,

E0541: r##"
An unknown meta item was used.

Erroneous code example:

```compile_fail,E0541
#[deprecated(
    since="1.0.0",
    // error: unknown meta item
    reason="Example invalid meta item. Should be 'note'")
]
fn deprecated_function() {}
```

Meta items are the key-value pairs inside of an attribute. The keys provided
must be one of the valid keys for the specified attribute.

To fix the problem, either remove the unknown meta item, or rename it if you
provided the wrong name.

In the erroneous code example above, the wrong name was provided, so changing
to a correct one it will fix the error. Example:

```
#[deprecated(
    since="1.0.0",
    note="This is a valid meta item for the deprecated attribute."
)]
fn deprecated_function() {}
```
"##,

E0550: r##"
More than one `deprecated` attribute has been put on an item.

Erroneous code example:

```compile_fail,E0550
#[deprecated(note = "because why not?")]
#[deprecated(note = "right?")] // error!
fn the_banished() {}
```

The `deprecated` attribute can only be present **once** on an item.

```
#[deprecated(note = "because why not, right?")]
fn the_banished() {} // ok!
```
"##,

E0551: r##"
An invalid meta-item was used inside an attribute.

Erroneous code example:

```compile_fail,E0551
#[deprecated(note)] // error!
fn i_am_deprecated() {}
```

Meta items are the key-value pairs inside of an attribute. To fix this issue,
you need to give a value to the `note` key. Example:

```
#[deprecated(note = "because")] // ok!
fn i_am_deprecated() {}
```
"##,

E0552: r##"
A unrecognized representation attribute was used.

Erroneous code example:

```compile_fail,E0552
#[repr(D)] // error: unrecognized representation hint
struct MyStruct {
    my_field: usize
}
```

You can use a `repr` attribute to tell the compiler how you want a struct or
enum to be laid out in memory.

Make sure you're using one of the supported options:

```
#[repr(C)] // ok!
struct MyStruct {
    my_field: usize
}
```

For more information about specifying representations, see the ["Alternative
Representations" section] of the Rustonomicon.

["Alternative Representations" section]: https://doc.rust-lang.org/nomicon/other-reprs.html
"##,

E0554: r##"
Feature attributes are only allowed on the nightly release channel. Stable or
beta compilers will not comply.

Example of erroneous code (on a stable compiler):

```ignore (depends on release channel)
#![feature(non_ascii_idents)] // error: `#![feature]` may not be used on the
                              //        stable release channel
```

If you need the feature, make sure to use a nightly release of the compiler
(but be warned that the feature may be removed or altered in the future).
"##,

E0556: r##"
The `feature` attribute was badly formed.

Erroneous code example:

```compile_fail,E0556
#![feature(foo_bar_baz, foo(bar), foo = "baz", foo)] // error!
#![feature] // error!
#![feature = "foo"] // error!
```

The `feature` attribute only accept a "feature flag" and can only be used on
nightly. Example:

```ignore (only works in nightly)
#![feature(flag)]
```
"##,

E0557: r##"
A feature attribute named a feature that has been removed.

Erroneous code example:

```compile_fail,E0557
#![feature(managed_boxes)] // error: feature has been removed
```

Delete the offending feature attribute.
"##,

E0559: r##"
An unknown field was specified into an enum's structure variant.

Erroneous code example:

```compile_fail,E0559
enum Field {
    Fool { x: u32 },
}

let s = Field::Fool { joke: 0 };
// error: struct variant `Field::Fool` has no field named `joke`
```

Verify you didn't misspell the field's name or that the field exists. Example:

```
enum Field {
    Fool { joke: u32 },
}

let s = Field::Fool { joke: 0 }; // ok!
```
"##,

E0560: r##"
An unknown field was specified into a structure.

Erroneous code example:

```compile_fail,E0560
struct Simba {
    mother: u32,
}

let s = Simba { mother: 1, father: 0 };
// error: structure `Simba` has no field named `father`
```

Verify you didn't misspell the field's name or that the field exists. Example:

```
struct Simba {
    mother: u32,
    father: u32,
}

let s = Simba { mother: 1, father: 0 }; // ok!
```
"##,

E0561: r##"
A non-ident or non-wildcard pattern has been used as a parameter of a function
pointer type.

Erroneous code example:

```compile_fail,E0561
type A1 = fn(mut param: u8); // error!
type A2 = fn(&param: u32); // error!
```

When using an alias over a function type, you cannot e.g. denote a parameter as
being mutable.

To fix the issue, remove patterns (`_` is allowed though). Example:

```
type A1 = fn(param: u8); // ok!
type A2 = fn(_: u32); // ok!
```

You can also omit the parameter name:

```
type A3 = fn(i16); // ok!
```
"##,

E0562: r##"
Abstract return types (written `impl Trait` for some trait `Trait`) are only
allowed as function and inherent impl return types.

Erroneous code example:

```compile_fail,E0562
fn main() {
    let count_to_ten: impl Iterator<Item=usize> = 0..10;
    // error: `impl Trait` not allowed outside of function and inherent method
    //        return types
    for i in count_to_ten {
        println!("{}", i);
    }
}
```

Make sure `impl Trait` only appears in return-type position.

```
fn count_to_n(n: usize) -> impl Iterator<Item=usize> {
    0..n
}

fn main() {
    for i in count_to_n(10) {  // ok!
        println!("{}", i);
    }
}
```

See [RFC 1522] for more details.

[RFC 1522]: https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md
"##,

E0565: r##"
A literal was used in a built-in attribute that doesn't support literals.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
#[inline("always")] // error: unsupported literal
pub fn something() {}
```

Literals in attributes are new and largely unsupported in built-in attributes.
Work to support literals where appropriate is ongoing. Try using an unquoted
name instead:

```
#[inline(always)]
pub fn something() {}
```
"##,

E0566: r##"
Conflicting representation hints have been used on a same item.

Erroneous code example:

```
#[repr(u32, u64)] // warning!
enum Repr { A }
```

In most cases (if not all), using just one representation hint is more than
enough. If you want to have a representation hint depending on the current
architecture, use `cfg_attr`. Example:

```
#[cfg_attr(linux, repr(u32))]
#[cfg_attr(not(linux), repr(u64))]
enum Repr { A }
```
"##,

E0567: r##"
Generics have been used on an auto trait.

Erroneous code example:

```compile_fail,E0567
#![feature(optin_builtin_traits)]

auto trait Generic<T> {} // error!

fn main() {}
```

Since an auto trait is implemented on all existing types, the
compiler would not be able to infer the types of the trait's generic
parameters.

To fix this issue, just remove the generics:

```
#![feature(optin_builtin_traits)]

auto trait Generic {} // ok!

fn main() {}
```
"##,

E0568: r##"
A super trait has been added to an auto trait.

Erroneous code example:

```compile_fail,E0568
#![feature(optin_builtin_traits)]

auto trait Bound : Copy {} // error!

fn main() {}
```

Since an auto trait is implemented on all existing types, adding a super trait
would filter out a lot of those types. In the current example, almost none of
all the existing types could implement `Bound` because very few of them have the
`Copy` trait.

To fix this issue, just remove the super trait:

```
#![feature(optin_builtin_traits)]

auto trait Bound {} // ok!

fn main() {}
```
"##,

E0569: r##"
If an impl has a generic parameter with the `#[may_dangle]` attribute, then
that impl must be declared as an `unsafe impl.

Erroneous code example:

```compile_fail,E0569
#![feature(dropck_eyepatch)]

struct Foo<X>(X);
impl<#[may_dangle] X> Drop for Foo<X> {
    fn drop(&mut self) { }
}
```

In this example, we are asserting that the destructor for `Foo` will not
access any data of type `X`, and require this assertion to be true for
overall safety in our program. The compiler does not currently attempt to
verify this assertion; therefore we must tag this `impl` as unsafe.
"##,

E0570: r##"
The requested ABI is unsupported by the current target.

The rust compiler maintains for each target a blacklist of ABIs unsupported on
that target. If an ABI is present in such a list this usually means that the
target / ABI combination is currently unsupported by llvm.

If necessary, you can circumvent this check using custom target specifications.
"##,

E0571: r##"
A `break` statement with an argument appeared in a non-`loop` loop.

Example of erroneous code:

```compile_fail,E0571
# let mut i = 1;
# fn satisfied(n: usize) -> bool { n % 23 == 0 }
let result = while true {
    if satisfied(i) {
        break 2*i; // error: `break` with value from a `while` loop
    }
    i += 1;
};
```

The `break` statement can take an argument (which will be the value of the loop
expression if the `break` statement is executed) in `loop` loops, but not
`for`, `while`, or `while let` loops.

Make sure `break value;` statements only occur in `loop` loops:

```
# let mut i = 1;
# fn satisfied(n: usize) -> bool { n % 23 == 0 }
let result = loop { // ok!
    if satisfied(i) {
        break 2*i;
    }
    i += 1;
};
```
"##,

E0572: r##"
A return statement was found outside of a function body.

Erroneous code example:

```compile_fail,E0572
const FOO: u32 = return 0; // error: return statement outside of function body

fn main() {}
```

To fix this issue, just remove the return keyword or move the expression into a
function. Example:

```
const FOO: u32 = 0;

fn some_fn() -> u32 {
    return FOO;
}

fn main() {
    some_fn();
}
```
"##,

E0573: r##"
Something other than a type has been used when one was expected.

Erroneous code examples:

```compile_fail,E0573
enum Dragon {
    Born,
}

fn oblivion() -> Dragon::Born { // error!
    Dragon::Born
}

const HOBBIT: u32 = 2;
impl HOBBIT {} // error!

enum Wizard {
    Gandalf,
    Saruman,
}

trait Isengard {
    fn wizard(_: Wizard::Saruman); // error!
}
```

In all these errors, a type was expected. For example, in the first error, if
we want to return the `Born` variant from the `Dragon` enum, we must set the
function to return the enum and not its variant:

```
enum Dragon {
    Born,
}

fn oblivion() -> Dragon { // ok!
    Dragon::Born
}
```

In the second error, you can't implement something on an item, only on types.
We would need to create a new type if we wanted to do something similar:

```
struct Hobbit(u32); // we create a new type

const HOBBIT: Hobbit = Hobbit(2);
impl Hobbit {} // ok!
```

In the third case, we tried to only expect one variant of the `Wizard` enum,
which is not possible. To make this work, we need to using pattern matching
over the `Wizard` enum:

```
enum Wizard {
    Gandalf,
    Saruman,
}

trait Isengard {
    fn wizard(w: Wizard) { // ok!
        match w {
            Wizard::Saruman => {
                // do something
            }
            _ => {} // ignore everything else
        }
    }
}
```
"##,

E0574: r##"
Something other than a struct, variant or union has been used when one was
expected.

Erroneous code example:

```compile_fail,E0574
mod Mordor {}

let sauron = Mordor { x: () }; // error!

enum Jak {
    Daxter { i: isize },
}

let eco = Jak::Daxter { i: 1 };
match eco {
    Jak { i } => {} // error!
}
```

In all these errors, a type was expected. For example, in the first error,
we tried to instantiate the `Mordor` module, which is impossible. If you want
to instantiate a type inside a module, you can do it as follow:

```
mod Mordor {
    pub struct TheRing {
        pub x: usize,
    }
}

let sauron = Mordor::TheRing { x: 1 }; // ok!
```

In the second error, we tried to bind the `Jak` enum directly, which is not
possible: you can only bind one of its variants. To do so:

```
enum Jak {
    Daxter { i: isize },
}

let eco = Jak::Daxter { i: 1 };
match eco {
    Jak::Daxter { i } => {} // ok!
}
```
"##,

E0575: r##"
Something other than a type or an associated type was given.

Erroneous code example:

```compile_fail,E0575
enum Rick { Morty }

let _: <u8 as Rick>::Morty; // error!

trait Age {
    type Empire;
    fn Mythology() {}
}

impl Age for u8 {
    type Empire = u16;
}

let _: <u8 as Age>::Mythology; // error!
```

In both cases, we're declaring a variable (called `_`) and we're giving it a
type. However, `<u8 as Rick>::Morty` and `<u8 as Age>::Mythology` aren't types,
therefore the compiler throws an error.

`<u8 as Rick>::Morty` is an enum variant, you cannot use a variant as a type,
you have to use the enum directly:

```
enum Rick { Morty }

let _: Rick; // ok!
```

`<u8 as Age>::Mythology` is a trait method, which is definitely not a type.
However, the `Age` trait provides an associated type `Empire` which can be
used as a type:

```
trait Age {
    type Empire;
    fn Mythology() {}
}

impl Age for u8 {
    type Empire = u16;
}

let _: <u8 as Age>::Empire; // ok!
```
"##,

E0576: r##"
An associated item wasn't found in the given type.

Erroneous code example:

```compile_fail,E0576
trait Hello {
    type Who;

    fn hello() -> <Self as Hello>::You; // error!
}
```

In this example, we tried to use the non-existent associated type `You` of the
`Hello` trait. To fix this error, use an existing associated type:

```
trait Hello {
    type Who;

    fn hello() -> <Self as Hello>::Who; // ok!
}
```
"##,

E0577: r##"
Something other than a module was found in visibility scope.

Erroneous code example:

```compile_fail,E0577,edition2018
pub struct Sea;

pub (in crate::Sea) struct Shark; // error!

fn main() {}
```

`Sea` is not a module, therefore it is invalid to use it in a visibility path.
To fix this error we need to ensure `Sea` is a module.

Please note that the visibility scope can only be applied on ancestors!

```edition2018
pub mod Sea {
    pub (in crate::Sea) struct Shark; // ok!
}

fn main() {}
```
"##,

E0578: r##"
A module cannot be found and therefore, the visibility cannot be determined.

Erroneous code example:

```compile_fail,E0578,edition2018
foo!();

pub (in ::Sea) struct Shark; // error!

fn main() {}
```

Because of the call to the `foo` macro, the compiler guesses that the missing
module could be inside it and fails because the macro definition cannot be
found.

To fix this error, please be sure that the module is in scope:

```edition2018
pub mod Sea {
    pub (in crate::Sea) struct Shark;
}

fn main() {}
```
"##,

E0579: r##"
When matching against an exclusive range, the compiler verifies that the range
is non-empty. Exclusive range patterns include the start point but not the end
point, so this is equivalent to requiring the start of the range to be less
than the end of the range.

Erroneous code example:

```compile_fail,E0579
#![feature(exclusive_range_pattern)]

fn main() {
    match 5u32 {
        // This range is ok, albeit pointless.
        1 .. 2 => {}
        // This range is empty, and the compiler can tell.
        5 .. 5 => {} // error!
    }
}
```
"##,

E0580: r##"
The `main` function was incorrectly declared.

Erroneous code example:

```compile_fail,E0580
fn main(x: i32) { // error: main function has wrong type
    println!("{}", x);
}
```

The `main` function prototype should never take arguments.
Example:

```
fn main() {
    // your code
}
```

If you want to get command-line arguments, use `std::env::args`. To exit with a
specified exit code, use `std::process::exit`.
"##,

E0581: r##"
In a `fn` type, a lifetime appears only in the return type,
and not in the arguments types.

Erroneous code example:

```compile_fail,E0581
fn main() {
    // Here, `'a` appears only in the return type:
    let x: for<'a> fn() -> &'a i32;
}
```

To fix this issue, either use the lifetime in the arguments, or use
`'static`. Example:

```
fn main() {
    // Here, `'a` appears only in the return type:
    let x: for<'a> fn(&'a i32) -> &'a i32;
    let y: fn() -> &'static i32;
}
```

Note: The examples above used to be (erroneously) accepted by the
compiler, but this was since corrected. See [issue #33685] for more
details.

[issue #33685]: https://github.com/rust-lang/rust/issues/33685
"##,

E0582: r##"
A lifetime appears only in an associated-type binding,
and not in the input types to the trait.

Erroneous code example:

```compile_fail,E0582
fn bar<F>(t: F)
    // No type can satisfy this requirement, since `'a` does not
    // appear in any of the input types (here, `i32`):
    where F: for<'a> Fn(i32) -> Option<&'a i32>
{
}

fn main() { }
```

To fix this issue, either use the lifetime in the inputs, or use
`'static`. Example:

```
fn bar<F, G>(t: F, u: G)
    where F: for<'a> Fn(&'a i32) -> Option<&'a i32>,
          G: Fn(i32) -> Option<&'static i32>,
{
}

fn main() { }
```

Note: The examples above used to be (erroneously) accepted by the
compiler, but this was since corrected. See [issue #33685] for more
details.

[issue #33685]: https://github.com/rust-lang/rust/issues/33685
"##,

E0583: r##"
A file wasn't found for an out-of-line module.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
mod file_that_doesnt_exist; // error: file not found for module

fn main() {}
```

Please be sure that a file corresponding to the module exists. If you
want to use a module named `file_that_doesnt_exist`, you need to have a file
named `file_that_doesnt_exist.rs` or `file_that_doesnt_exist/mod.rs` in the
same directory.
"##,

E0584: r##"
A doc comment that is not attached to anything has been encountered.

Erroneous code example:

```compile_fail,E0584
trait Island {
    fn lost();

    /// I'm lost!
}
```

A little reminder: a doc comment has to be placed before the item it's supposed
to document. So if you want to document the `Island` trait, you need to put a
doc comment before it, not inside it. Same goes for the `lost` method: the doc
comment needs to be before it:

```
/// I'm THE island!
trait Island {
    /// I'm lost!
    fn lost();
}
```
"##,

E0585: r##"
A documentation comment that doesn't document anything was found.

Erroneous code example:

```compile_fail,E0585
fn main() {
    // The following doc comment will fail:
    /// This is a useless doc comment!
}
```

Documentation comments need to be followed by items, including functions,
types, modules, etc. Examples:

```
/// I'm documenting the following struct:
struct Foo;

/// I'm documenting the following function:
fn foo() {}
```
"##,

E0586: r##"
An inclusive range was used with no end.

Erroneous code example:

```compile_fail,E0586
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..=]; // error: inclusive range was used with no end
}
```

An inclusive range needs an end in order to *include* it. If you just need a
start and no end, use a non-inclusive range (with `..`):

```
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..]; // ok!
}
```

Or put an end to your inclusive range:

```
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..=3]; // ok!
}
```
"##,

E0587: r##"
A type has both `packed` and `align` representation hints.

Erroneous code example:

```compile_fail,E0587
#[repr(packed, align(8))] // error!
struct Umbrella(i32);
```

You cannot use `packed` and `align` hints on a same type. If you want to pack a
type to a given size, you should provide a size to packed:

```
#[repr(packed)] // ok!
struct Umbrella(i32);
```
"##,

E0588: r##"
A type with `packed` representation hint has a field with `align`
representation hint.

Erroneous code example:

```compile_fail,E0588
#[repr(align(16))]
struct Aligned(i32);

#[repr(packed)] // error!
struct Packed(Aligned);
```

Just like you cannot have both `align` and `packed` representation hints on a
same type, a `packed` type cannot contain another type with the `align`
representation hint. However, you can do the opposite:

```
#[repr(packed)]
struct Packed(i32);

#[repr(align(16))] // ok!
struct Aligned(Packed);
```
"##,

E0589: r##"
The value of `N` that was specified for `repr(align(N))` was not a power
of two, or was greater than 2^29.

```compile_fail,E0589
#[repr(align(15))] // error: invalid `repr(align)` attribute: not a power of two
enum Foo {
    Bar(u64),
}
```
"##,

E0590: r##"
`break` or `continue` must include a label when used in the condition of a
`while` loop.

Example of erroneous code:

```compile_fail
while break {}
```

To fix this, add a label specifying which loop is being broken out of:
```
'foo: while break 'foo {}
```
"##,

E0591: r##"
Per [RFC 401][rfc401], if you have a function declaration `foo`:

```
// For the purposes of this explanation, all of these
// different kinds of `fn` declarations are equivalent:
struct S;
fn foo(x: S) { /* ... */ }
# #[cfg(for_demonstration_only)]
extern "C" { fn foo(x: S); }
# #[cfg(for_demonstration_only)]
impl S { fn foo(self) { /* ... */ } }
```

the type of `foo` is **not** `fn(S)`, as one might expect.
Rather, it is a unique, zero-sized marker type written here as `typeof(foo)`.
However, `typeof(foo)` can be _coerced_ to a function pointer `fn(S)`,
so you rarely notice this:

```
# struct S;
# fn foo(_: S) {}
let x: fn(S) = foo; // OK, coerces
```

The reason that this matter is that the type `fn(S)` is not specific to
any particular function: it's a function _pointer_. So calling `x()` results
in a virtual call, whereas `foo()` is statically dispatched, because the type
of `foo` tells us precisely what function is being called.

As noted above, coercions mean that most code doesn't have to be
concerned with this distinction. However, you can tell the difference
when using **transmute** to convert a fn item into a fn pointer.

This is sometimes done as part of an FFI:

```compile_fail,E0591
extern "C" fn foo(userdata: Box<i32>) {
    /* ... */
}

# fn callback(_: extern "C" fn(*mut i32)) {}
# use std::mem::transmute;
# unsafe {
let f: extern "C" fn(*mut i32) = transmute(foo);
callback(f);
# }
```

Here, transmute is being used to convert the types of the fn arguments.
This pattern is incorrect because, because the type of `foo` is a function
**item** (`typeof(foo)`), which is zero-sized, and the target type (`fn()`)
is a function pointer, which is not zero-sized.
This pattern should be rewritten. There are a few possible ways to do this:

- change the original fn declaration to match the expected signature,
  and do the cast in the fn body (the preferred option)
- cast the fn item fo a fn pointer before calling transmute, as shown here:

    ```
    # extern "C" fn foo(_: Box<i32>) {}
    # use std::mem::transmute;
    # unsafe {
    let f: extern "C" fn(*mut i32) = transmute(foo as extern "C" fn(_));
    let f: extern "C" fn(*mut i32) = transmute(foo as usize); // works too
    # }
    ```

The same applies to transmutes to `*mut fn()`, which were observed in practice.
Note though that use of this type is generally incorrect.
The intention is typically to describe a function pointer, but just `fn()`
alone suffices for that. `*mut fn()` is a pointer to a fn pointer.
(Since these values are typically just passed to C code, however, this rarely
makes a difference in practice.)

[rfc401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
"##,

E0592: r##"
This error occurs when you defined methods or associated functions with same
name.

Erroneous code example:

```compile_fail,E0592
struct Foo;

impl Foo {
    fn bar() {} // previous definition here
}

impl Foo {
    fn bar() {} // duplicate definition here
}
```

A similar error is E0201. The difference is whether there is one declaration
block or not. To avoid this error, you must give each `fn` a unique name.

```
struct Foo;

impl Foo {
    fn bar() {}
}

impl Foo {
    fn baz() {} // define with different name
}
```
"##,

E0593: r##"
You tried to supply an `Fn`-based type with an incorrect number of arguments
than what was expected.

Erroneous code example:

```compile_fail,E0593
fn foo<F: Fn()>(x: F) { }

fn main() {
    // [E0593] closure takes 1 argument but 0 arguments are required
    foo(|y| { });
}
```
"##,

E0595: r##"
#### Note: this error code is no longer emitted by the compiler.

Closures cannot mutate immutable captured variables.

Erroneous code example:

```compile_fail,E0594
let x = 3; // error: closure cannot assign to immutable local variable `x`
let mut c = || { x += 1 };
```

Make the variable binding mutable:

```
let mut x = 3; // ok!
let mut c = || { x += 1 };
```
"##,

E0596: r##"
This error occurs because you tried to mutably borrow a non-mutable variable.

Erroneous code example:

```compile_fail,E0596
let x = 1;
let y = &mut x; // error: cannot borrow mutably
```

In here, `x` isn't mutable, so when we try to mutably borrow it in `y`, it
fails. To fix this error, you need to make `x` mutable:

```
let mut x = 1;
let y = &mut x; // ok!
```
"##,

E0597: r##"
This error occurs because a value was dropped while it was still borrowed

Erroneous code example:

```compile_fail,E0597
struct Foo<'a> {
    x: Option<&'a u32>,
}

let mut x = Foo { x: None };
{
    let y = 0;
    x.x = Some(&y); // error: `y` does not live long enough
}
println!("{:?}", x.x);
```

In here, `y` is dropped at the end of the inner scope, but it is borrowed by
`x` until the `println`. To fix the previous example, just remove the scope
so that `y` isn't dropped until after the println

```
struct Foo<'a> {
    x: Option<&'a u32>,
}

let mut x = Foo { x: None };

let y = 0;
x.x = Some(&y);

println!("{:?}", x.x);
```
"##,

E0599: r##"
This error occurs when a method is used on a type which doesn't implement it:

Erroneous code example:

```compile_fail,E0599
struct Mouth;

let x = Mouth;
x.chocolate(); // error: no method named `chocolate` found for type `Mouth`
               //        in the current scope
```
"##,

E0600: r##"
An unary operator was used on a type which doesn't implement it.

Example of erroneous code:

```compile_fail,E0600
enum Question {
    Yes,
    No,
}

!Question::Yes; // error: cannot apply unary operator `!` to type `Question`
```

In this case, `Question` would need to implement the `std::ops::Not` trait in
order to be able to use `!` on it. Let's implement it:

```
use std::ops::Not;

enum Question {
    Yes,
    No,
}

// We implement the `Not` trait on the enum.
impl Not for Question {
    type Output = bool;

    fn not(self) -> bool {
        match self {
            Question::Yes => false, // If the `Answer` is `Yes`, then it
                                    // returns false.
            Question::No => true, // And here we do the opposite.
        }
    }
}

assert_eq!(!Question::Yes, false);
assert_eq!(!Question::No, true);
```
"##,

E0601: r##"
No `main` function was found in a binary crate. To fix this error, add a
`main` function. For example:

```
fn main() {
    // Your program will start here.
    println!("Hello world!");
}
```

If you don't know the basics of Rust, you can go look to the Rust Book to get
started: https://doc.rust-lang.org/book/
"##,

E0602: r##"
An unknown lint was used on the command line.

Erroneous example:

```sh
rustc -D bogus omse_file.rs
```

Maybe you just misspelled the lint name or the lint doesn't exist anymore.
Either way, try to update/remove it in order to fix the error.
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

E0604: r##"
A cast to `char` was attempted on a type other than `u8`.

Erroneous code example:

```compile_fail,E0604
0u32 as char; // error: only `u8` can be cast as `char`, not `u32`
```

As the error message indicates, only `u8` can be cast into `char`. Example:

```
let c = 86u8 as char; // ok!
assert_eq!(c, 'V');
```

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
"##,

E0605: r##"
An invalid cast was attempted.

Erroneous code examples:

```compile_fail,E0605
let x = 0u8;
x as Vec<u8>; // error: non-primitive cast: `u8` as `std::vec::Vec<u8>`

// Another example

let v = core::ptr::null::<u8>(); // So here, `v` is a `*const u8`.
v as &u8; // error: non-primitive cast: `*const u8` as `&u8`
```

Only primitive types can be cast into each other. Examples:

```
let x = 0u8;
x as u32; // ok!

let v = core::ptr::null::<u8>();
v as *const i8; // ok!
```

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
"##,

E0606: r##"
An incompatible cast was attempted.

Erroneous code example:

```compile_fail,E0606
let x = &0u8; // Here, `x` is a `&u8`.
let y: u32 = x as u32; // error: casting `&u8` as `u32` is invalid
```

When casting, keep in mind that only primitive types can be cast into each
other. Example:

```
let x = &0u8;
let y: u32 = *x as u32; // We dereference it first and then cast it.
```

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
"##,

E0607: r##"
A cast between a thin and a fat pointer was attempted.

Erroneous code example:

```compile_fail,E0607
let v = core::ptr::null::<u8>();
v as *const [u8];
```

First: what are thin and fat pointers?

Thin pointers are "simple" pointers: they are purely a reference to a memory
address.

Fat pointers are pointers referencing Dynamically Sized Types (also called DST).
DST don't have a statically known size, therefore they can only exist behind
some kind of pointers that contain additional information. Slices and trait
objects are DSTs. In the case of slices, the additional information the fat
pointer holds is their size.

To fix this error, don't try to cast directly between thin and fat pointers.

For more information about casts, take a look at the Type cast section in
[The Reference Book][1].

[1]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
"##,

E0608: r##"
An attempt to index into a type which doesn't implement the `std::ops::Index`
trait was performed.

Erroneous code example:

```compile_fail,E0608
0u8[2]; // error: cannot index into a value of type `u8`
```

To be able to index into a type it needs to implement the `std::ops::Index`
trait. Example:

```
let v: Vec<u8> = vec![0, 1, 2, 3];

// The `Vec` type implements the `Index` trait so you can do:
println!("{}", v[2]);
```
"##,

E0609: r##"
Attempted to access a non-existent field in a struct.

Erroneous code example:

```compile_fail,E0609
struct StructWithFields {
    x: u32,
}

let s = StructWithFields { x: 0 };
println!("{}", s.foo); // error: no field `foo` on type `StructWithFields`
```

To fix this error, check that you didn't misspell the field's name or that the
field actually exists. Example:

```
struct StructWithFields {
    x: u32,
}

let s = StructWithFields { x: 0 };
println!("{}", s.x); // ok!
```
"##,

E0610: r##"
Attempted to access a field on a primitive type.

Erroneous code example:

```compile_fail,E0610
let x: u32 = 0;
println!("{}", x.foo); // error: `{integer}` is a primitive type, therefore
                       //        doesn't have fields
```

Primitive types are the most basic types available in Rust and don't have
fields. To access data via named fields, struct types are used. Example:

```
// We declare struct called `Foo` containing two fields:
struct Foo {
    x: u32,
    y: i64,
}

// We create an instance of this struct:
let variable = Foo { x: 0, y: -12 };
// And we can now access its fields:
println!("x: {}, y: {}", variable.x, variable.y);
```

For more information about primitives and structs, take a look at The Book:
https://doc.rust-lang.org/book/ch03-02-data-types.html
https://doc.rust-lang.org/book/ch05-00-structs.html
"##,

E0614: r##"
Attempted to dereference a variable which cannot be dereferenced.

Erroneous code example:

```compile_fail,E0614
let y = 0u32;
*y; // error: type `u32` cannot be dereferenced
```

Only types implementing `std::ops::Deref` can be dereferenced (such as `&T`).
Example:

```
let y = 0u32;
let x = &y;
// So here, `x` is a `&u32`, so we can dereference it:
*x; // ok!
```
"##,

E0615: r##"
Attempted to access a method like a field.

Erroneous code example:

```compile_fail,E0615
struct Foo {
    x: u32,
}

impl Foo {
    fn method(&self) {}
}

let f = Foo { x: 0 };
f.method; // error: attempted to take value of method `method` on type `Foo`
```

If you want to use a method, add `()` after it:

```
# struct Foo { x: u32 }
# impl Foo { fn method(&self) {} }
# let f = Foo { x: 0 };
f.method();
```

However, if you wanted to access a field of a struct check that the field name
is spelled correctly. Example:

```
# struct Foo { x: u32 }
# impl Foo { fn method(&self) {} }
# let f = Foo { x: 0 };
println!("{}", f.x);
```
"##,

E0616: r##"
Attempted to access a private field on a struct.

Erroneous code example:

```compile_fail,E0616
mod some_module {
    pub struct Foo {
        x: u32, // So `x` is private in here.
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }
    }
}

let f = some_module::Foo::new();
println!("{}", f.x); // error: field `x` of struct `some_module::Foo` is private
```

If you want to access this field, you have two options:

1) Set the field public:

```
mod some_module {
    pub struct Foo {
        pub x: u32, // `x` is now public.
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }
    }
}

let f = some_module::Foo::new();
println!("{}", f.x); // ok!
```

2) Add a getter function:

```
mod some_module {
    pub struct Foo {
        x: u32, // So `x` is still private in here.
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }

        // We create the getter function here:
        pub fn get_x(&self) -> &u32 { &self.x }
    }
}

let f = some_module::Foo::new();
println!("{}", f.get_x()); // ok!
```
"##,

E0617: r##"
Attempted to pass an invalid type of variable into a variadic function.

Erroneous code example:

```compile_fail,E0617
extern {
    fn printf(c: *const i8, ...);
}

unsafe {
    printf(::std::ptr::null(), 0f32);
    // error: cannot pass an `f32` to variadic function, cast to `c_double`
}
```

Certain Rust types must be cast before passing them to a variadic function,
because of arcane ABI rules dictated by the C standard. To fix the error,
cast the value to the type specified by the error message (which you may need
to import from `std::os::raw`).
"##,

E0618: r##"
Attempted to call something which isn't a function nor a method.

Erroneous code examples:

```compile_fail,E0618
enum X {
    Entry,
}

X::Entry(); // error: expected function, tuple struct or tuple variant,
            // found `X::Entry`

// Or even simpler:
let x = 0i32;
x(); // error: expected function, tuple struct or tuple variant, found `i32`
```

Only functions and methods can be called using `()`. Example:

```
// We declare a function:
fn i_am_a_function() {}

// And we call it:
i_am_a_function();
```
"##,

E0619: r##"
#### Note: this error code is no longer emitted by the compiler.
The type-checker needed to know the type of an expression, but that type had not
yet been inferred.

Erroneous code example:

```compile_fail
let mut x = vec![];
match x.pop() {
    Some(v) => {
        // Here, the type of `v` is not (yet) known, so we
        // cannot resolve this method call:
        v.to_uppercase(); // error: the type of this value must be known in
                          //        this context
    }
    None => {}
}
```

Type inference typically proceeds from the top of the function to the bottom,
figuring out types as it goes. In some cases -- notably method calls and
overloadable operators like `*` -- the type checker may not have enough
information *yet* to make progress. This can be true even if the rest of the
function provides enough context (because the type-checker hasn't looked that
far ahead yet). In this case, type annotations can be used to help it along.

To fix this error, just specify the type of the variable. Example:

```
let mut x: Vec<String> = vec![]; // We precise the type of the vec elements.
match x.pop() {
    Some(v) => {
        v.to_uppercase(); // Since rustc now knows the type of the vec elements,
                          // we can use `v`'s methods.
    }
    None => {}
}
```
"##,

E0620: r##"
A cast to an unsized type was attempted.

Erroneous code example:

```compile_fail,E0620
let x = &[1_usize, 2] as [usize]; // error: cast to unsized type: `&[usize; 2]`
                                  //        as `[usize]`
```

In Rust, some types don't have a known size at compile-time. For example, in a
slice type like `[u32]`, the number of elements is not known at compile-time and
hence the overall size cannot be computed. As a result, such types can only be
manipulated through a reference (e.g., `&T` or `&mut T`) or other pointer-type
(e.g., `Box` or `Rc`). Try casting to a reference instead:

```
let x = &[1_usize, 2] as &[usize]; // ok!
```
"##,

E0621: r##"
This error code indicates a mismatch between the lifetimes appearing in the
function signature (i.e., the parameter types and the return type) and the
data-flow found in the function body.

Erroneous code example:

```compile_fail,E0621
fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32 { // error: explicit lifetime
                                             //        required in the type of
                                             //        `y`
    if x > y { x } else { y }
}
```

In the code above, the function is returning data borrowed from either `x` or
`y`, but the `'a` annotation indicates that it is returning data only from `x`.
To fix the error, the signature and the body must be made to match. Typically,
this is done by updating the function signature. So, in this case, we change
the type of `y` to `&'a i32`, like so:

```
fn foo<'a>(x: &'a i32, y: &'a i32) -> &'a i32 {
    if x > y { x } else { y }
}
```

Now the signature indicates that the function data borrowed from either `x` or
`y`. Alternatively, you could change the body to not return data from `y`:

```
fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32 {
    x
}
```
"##,

E0622: r##"
An intrinsic was declared without being a function.

Erroneous code example:

```compile_fail,E0622
#![feature(intrinsics)]
extern "rust-intrinsic" {
    pub static breakpoint : unsafe extern "rust-intrinsic" fn();
    // error: intrinsic must be a function
}

fn main() { unsafe { breakpoint(); } }
```

An intrinsic is a function available for use in a given programming language
whose implementation is handled specially by the compiler. In order to fix this
error, just declare a function.
"##,

E0624: r##"
A private item was used outside of its scope.

Erroneous code example:

```compile_fail,E0624
mod inner {
    pub struct Foo;

    impl Foo {
        fn method(&self) {}
    }
}

let foo = inner::Foo;
foo.method(); // error: method `method` is private
```

Two possibilities are available to solve this issue:

1. Only use the item in the scope it has been defined:

```
mod inner {
    pub struct Foo;

    impl Foo {
        fn method(&self) {}
    }

    pub fn call_method(foo: &Foo) { // We create a public function.
        foo.method(); // Which calls the item.
    }
}

let foo = inner::Foo;
inner::call_method(&foo); // And since the function is public, we can call the
                          // method through it.
```

2. Make the item public:

```
mod inner {
    pub struct Foo;

    impl Foo {
        pub fn method(&self) {} // It's now public.
    }
}

let foo = inner::Foo;
foo.method(); // Ok!
```
"##,

E0626: r##"
This error occurs because a borrow in a generator persists across a
yield point.

Erroneous code example:

```compile_fail,E0626
# #![feature(generators, generator_trait, pin)]
# use std::ops::Generator;
# use std::pin::Pin;
let mut b = || {
    let a = &String::new(); // <-- This borrow...
    yield (); // ...is still in scope here, when the yield occurs.
    println!("{}", a);
};
Pin::new(&mut b).resume();
```

At present, it is not permitted to have a yield that occurs while a
borrow is still in scope. To resolve this error, the borrow must
either be "contained" to a smaller scope that does not overlap the
yield or else eliminated in another way. So, for example, we might
resolve the previous example by removing the borrow and just storing
the integer by value:

```
# #![feature(generators, generator_trait, pin)]
# use std::ops::Generator;
# use std::pin::Pin;
let mut b = || {
    let a = 3;
    yield ();
    println!("{}", a);
};
Pin::new(&mut b).resume();
```

This is a very simple case, of course. In more complex cases, we may
wish to have more than one reference to the value that was borrowed --
in those cases, something like the `Rc` or `Arc` types may be useful.

This error also frequently arises with iteration:

```compile_fail,E0626
# #![feature(generators, generator_trait, pin)]
# use std::ops::Generator;
# use std::pin::Pin;
let mut b = || {
  let v = vec![1,2,3];
  for &x in &v { // <-- borrow of `v` is still in scope...
    yield x; // ...when this yield occurs.
  }
};
Pin::new(&mut b).resume();
```

Such cases can sometimes be resolved by iterating "by value" (or using
`into_iter()`) to avoid borrowing:

```
# #![feature(generators, generator_trait, pin)]
# use std::ops::Generator;
# use std::pin::Pin;
let mut b = || {
  let v = vec![1,2,3];
  for x in v { // <-- Take ownership of the values instead!
    yield x; // <-- Now yield is OK.
  }
};
Pin::new(&mut b).resume();
```

If taking ownership is not an option, using indices can work too:

```
# #![feature(generators, generator_trait, pin)]
# use std::ops::Generator;
# use std::pin::Pin;
let mut b = || {
  let v = vec![1,2,3];
  let len = v.len(); // (*)
  for i in 0..len {
    let x = v[i]; // (*)
    yield x; // <-- Now yield is OK.
  }
};
Pin::new(&mut b).resume();

// (*) -- Unfortunately, these temporaries are currently required.
// See <https://github.com/rust-lang/rust/issues/43122>.
```
"##,

E0633: r##"
The `unwind` attribute was malformed.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
#[unwind()] // error: expected one argument
pub extern fn something() {}

fn main() {}
```

The `#[unwind]` attribute should be used as follows:

- `#[unwind(aborts)]` -- specifies that if a non-Rust ABI function
  should abort the process if it attempts to unwind. This is the safer
  and preferred option.

- `#[unwind(allowed)]` -- specifies that a non-Rust ABI function
  should be allowed to unwind. This can easily result in Undefined
  Behavior (UB), so be careful.

NB. The default behavior here is "allowed", but this is unspecified
and likely to change in the future.

"##,

E0635: r##"
The `#![feature]` attribute specified an unknown feature.

Erroneous code example:

```compile_fail,E0635
#![feature(nonexistent_rust_feature)] // error: unknown feature
```

"##,

E0636: r##"
A `#![feature]` attribute was declared multiple times.

Erroneous code example:

```compile_fail,E0636
#![allow(stable_features)]
#![feature(rust1)]
#![feature(rust1)] // error: the feature `rust1` has already been declared
```

"##,

E0638: r##"
This error indicates that the struct, enum or enum variant must be matched
non-exhaustively as it has been marked as `non_exhaustive`.

When applied within a crate, downstream users of the crate will need to use the
`_` pattern when matching enums and use the `..` pattern when matching structs.
Downstream crates cannot match against non-exhaustive enum variants.

For example, in the below example, since the enum is marked as
`non_exhaustive`, it is required that downstream crates match non-exhaustively
on it.

```rust,ignore (pseudo-Rust)
use std::error::Error as StdError;

#[non_exhaustive] pub enum Error {
   Message(String),
   Other,
}

impl StdError for Error {
   fn description(&self) -> &str {
        // This will not error, despite being marked as non_exhaustive, as this
        // enum is defined within the current crate, it can be matched
        // exhaustively.
        match *self {
           Message(ref s) => s,
           Other => "other or unknown error",
        }
   }
}
```

An example of matching non-exhaustively on the above enum is provided below:

```rust,ignore (pseudo-Rust)
use mycrate::Error;

// This will not error as the non_exhaustive Error enum has been matched with a
// wildcard.
match error {
   Message(ref s) => ...,
   Other => ...,
   _ => ...,
}
```

Similarly, for structs, match with `..` to avoid this error.
"##,

E0639: r##"
This error indicates that the struct, enum or enum variant cannot be
instantiated from outside of the defining crate as it has been marked
as `non_exhaustive` and as such more fields/variants may be added in
future that could cause adverse side effects for this code.

It is recommended that you look for a `new` function or equivalent in the
crate's documentation.
"##,

E0642: r##"
Trait methods currently cannot take patterns as arguments.

Example of erroneous code:

```compile_fail,E0642
trait Foo {
    fn foo((x, y): (i32, i32)); // error: patterns aren't allowed
                                //        in trait methods
}
```

You can instead use a single name for the argument:

```
trait Foo {
    fn foo(x_and_y: (i32, i32)); // ok!
}
```
"##,

E0643: r##"
This error indicates that there is a mismatch between generic parameters and
impl Trait parameters in a trait declaration versus its impl.

```compile_fail,E0643
trait Foo {
    fn foo(&self, _: &impl Iterator);
}
impl Foo for () {
    fn foo<U: Iterator>(&self, _: &U) { } // error method `foo` has incompatible
                                          // signature for trait
}
```
"##,

E0644: r##"
A closure or generator was constructed that references its own type.

Erroneous example:

```compile-fail,E0644
fn fix<F>(f: &F)
  where F: Fn(&F)
{
  f(&f);
}

fn main() {
  fix(&|y| {
    // Here, when `x` is called, the parameter `y` is equal to `x`.
  });
}
```

Rust does not permit a closure to directly reference its own type,
either through an argument (as in the example above) or by capturing
itself through its environment. This restriction helps keep closure
inference tractable.

The easiest fix is to rewrite your closure into a top-level function,
or into a method. In some cases, you may also be able to have your
closure call itself by capturing a `&Fn()` object or `fn()` pointer
that refers to itself. That is permitting, since the closure would be
invoking itself via a virtual call, and hence does not directly
reference its own *type*.

"##,

E0646: r##"
It is not possible to define `main` with a where clause.
Erroneous code example:

```compile_fail,E0646
fn main() where i32: Copy { // error: main function is not allowed to have
                            // a where clause
}
```
"##,

E0647: r##"
It is not possible to define `start` with a where clause.
Erroneous code example:

```compile_fail,E0647
#![feature(start)]

#[start]
fn start(_: isize, _: *const *const u8) -> isize where (): Copy {
    //^ error: start function is not allowed to have a where clause
    0
}
```
"##,

E0648: r##"
`export_name` attributes may not contain null characters (`\0`).

```compile_fail,E0648
#[export_name="\0foo"] // error: `export_name` may not contain null characters
pub fn bar() {}
```
"##,

E0658: r##"
An unstable feature was used.

Erroneous code example:

```compile_fail,E658
#[repr(u128)] // error: use of unstable library feature 'repr128'
enum Foo {
    Bar(u64),
}
```

If you're using a stable or a beta version of rustc, you won't be able to use
any unstable features. In order to do so, please switch to a nightly version of
rustc (by using rustup).

If you're using a nightly version of rustc, just add the corresponding feature
to be able to use it:

```
#![feature(repr128)]

#[repr(u128)] // ok!
enum Foo {
    Bar(u64),
}
```
"##,

E0659: r##"
An item usage is ambiguous.

Erroneous code example:

```compile_fail,edition2018,E0659
pub mod moon {
    pub fn foo() {}
}

pub mod earth {
    pub fn foo() {}
}

mod collider {
    pub use crate::moon::*;
    pub use crate::earth::*;
}

fn main() {
    crate::collider::foo(); // ERROR: `foo` is ambiguous
}
```

This error generally appears when two items with the same name are imported into
a module. Here, the `foo` functions are imported and reexported from the
`collider` module and therefore, when we're using `collider::foo()`, both
functions collide.

To solve this error, the best solution is generally to keep the path before the
item when using it. Example:

```edition2018
pub mod moon {
    pub fn foo() {}
}

pub mod earth {
    pub fn foo() {}
}

mod collider {
    pub use crate::moon;
    pub use crate::earth;
}

fn main() {
    crate::collider::moon::foo(); // ok!
    crate::collider::earth::foo(); // ok!
}
```
"##,

E0660: r##"
The argument to the `asm` macro is not well-formed.

Erroneous code example:

```compile_fail,E0660
asm!("nop" "nop");
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0661: r##"
An invalid syntax was passed to the second argument of an `asm` macro line.

Erroneous code example:

```compile_fail,E0661
let a;
asm!("nop" : "r"(a));
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0662: r##"
An invalid input operand constraint was passed to the `asm` macro (third line).

Erroneous code example:

```compile_fail,E0662
asm!("xor %eax, %eax"
     :
     : "=test"("a")
    );
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0663: r##"
An invalid input operand constraint was passed to the `asm` macro (third line).

Erroneous code example:

```compile_fail,E0663
asm!("xor %eax, %eax"
     :
     : "+test"("a")
    );
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0664: r##"
A clobber was surrounded by braces in the `asm` macro.

Erroneous code example:

```compile_fail,E0664
asm!("mov $$0x200, %eax"
     :
     :
     : "{eax}"
    );
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0665: r##"
The `Default` trait was derived on an enum.

Erroneous code example:

```compile_fail,E0665
#[derive(Default)]
enum Food {
    Sweet,
    Salty,
}
```

The `Default` cannot be derived on an enum for the simple reason that the
compiler doesn't know which value to pick by default whereas it can for a
struct as long as all its fields implement the `Default` trait as well.

If you still want to implement `Default` on your enum, you'll have to do it "by
hand":

```
enum Food {
    Sweet,
    Salty,
}

impl Default for Food {
    fn default() -> Food {
        Food::Sweet
    }
}
```
"##,

E0666: r##"
`impl Trait` types cannot appear nested in the
generic arguments of other `impl Trait` types.

Example of erroneous code:

```compile_fail,E0666
trait MyGenericTrait<T> {}
trait MyInnerTrait {}

fn foo(bar: impl MyGenericTrait<impl MyInnerTrait>) {}
```

Type parameters for `impl Trait` types must be
explicitly defined as named generic parameters:

```
trait MyGenericTrait<T> {}
trait MyInnerTrait {}

fn foo<T: MyInnerTrait>(bar: impl MyGenericTrait<T>) {}
```
"##,

E0668: r##"
Malformed inline assembly rejected by LLVM.

LLVM checks the validity of the constraints and the assembly string passed to
it. This error implies that LLVM seems something wrong with the inline
assembly call.

In particular, it can happen if you forgot the closing bracket of a register
constraint (see issue #51430):
```ignore (error-emitted-at-codegen-which-cannot-be-handled-by-compile_fail)
#![feature(asm)]

fn main() {
    let rax: u64;
    unsafe {
        asm!("" :"={rax"(rax));
        println!("Accumulator is: {}", rax);
    }
}
```
"##,

E0669: r##"
Cannot convert inline assembly operand to a single LLVM value.

This error usually happens when trying to pass in a value to an input inline
assembly operand that is actually a pair of values. In particular, this can
happen when trying to pass in a slice, for instance a `&str`. In Rust, these
values are represented internally as a pair of values, the pointer and its
length. When passed as an input operand, this pair of values can not be
coerced into a register and thus we must fail with an error.
"##,

E0670: r##"
Rust 2015 does not permit the use of `async fn`.

Example of erroneous code:

```compile_fail,E0670
async fn foo() {}
```

Switch to the Rust 2018 edition to use `async fn`.
"##,

E0671: r##"
#### Note: this error code is no longer emitted by the compiler.

Const parameters cannot depend on type parameters.
The following is therefore invalid:

```compile_fail,E0741
#![feature(const_generics)]

fn const_id<T, const N: T>() -> T { // error
    N
}
```
"##,

E0689: r##"
This error indicates that the numeric value for the method being passed exists
but the type of the numeric value or binding could not be identified.

The error happens on numeric literals:

```compile_fail,E0689
2.0.neg();
```

and on numeric bindings without an identified concrete type:

```compile_fail,E0689
let x = 2.0;
x.neg();  // same error as above
```

Because of this, you must give the numeric literal or binding a type:

```
use std::ops::Neg;

let _ = 2.0_f32.neg();
let x: f32 = 2.0;
let _ = x.neg();
let _ = (2.0 as f32).neg();
```
"##,

E0690: r##"
A struct with the representation hint `repr(transparent)` had zero or more than
one fields that were not guaranteed to be zero-sized.

Erroneous code example:

```compile_fail,E0690
#[repr(transparent)]
struct LengthWithUnit<U> { // error: transparent struct needs exactly one
    value: f32,            //        non-zero-sized field, but has 2
    unit: U,
}
```

Because transparent structs are represented exactly like one of their fields at
run time, said field must be uniquely determined. If there is no field, or if
there are multiple fields, it is not clear how the struct should be represented.
Note that fields of zero-typed types (e.g., `PhantomData`) can also exist
alongside the field that contains the actual data, they do not count for this
error. When generic types are involved (as in the above example), an error is
reported because the type parameter could be non-zero-sized.

To combine `repr(transparent)` with type parameters, `PhantomData` may be
useful:

```
use std::marker::PhantomData;

#[repr(transparent)]
struct LengthWithUnit<U> {
    value: f32,
    unit: PhantomData<U>,
}
```
"##,

E0691: r##"
A struct, enum, or union with the `repr(transparent)` representation hint
contains a zero-sized field that requires non-trivial alignment.

Erroneous code example:

```compile_fail,E0691
#![feature(repr_align)]

#[repr(align(32))]
struct ForceAlign32;

#[repr(transparent)]
struct Wrapper(f32, ForceAlign32); // error: zero-sized field in transparent
                                   //        struct has alignment larger than 1
```

A transparent struct, enum, or union is supposed to be represented exactly like
the piece of data it contains. Zero-sized fields with different alignment
requirements potentially conflict with this property. In the example above,
`Wrapper` would have to be aligned to 32 bytes even though `f32` has a smaller
alignment requirement.

Consider removing the over-aligned zero-sized field:

```
#[repr(transparent)]
struct Wrapper(f32);
```

Alternatively, `PhantomData<T>` has alignment 1 for all `T`, so you can use it
if you need to keep the field for some reason:

```
#![feature(repr_align)]

use std::marker::PhantomData;

#[repr(align(32))]
struct ForceAlign32;

#[repr(transparent)]
struct Wrapper(f32, PhantomData<ForceAlign32>);
```

Note that empty arrays `[T; 0]` have the same alignment requirement as the
element type `T`. Also note that the error is conservatively reported even when
the alignment of the zero-sized type is less than or equal to the data field's
alignment.
"##,

E0692: r##"
A `repr(transparent)` type was also annotated with other, incompatible
representation hints.

Erroneous code example:

```compile_fail,E0692
#[repr(transparent, C)] // error: incompatible representation hints
struct Grams(f32);
```

A type annotated as `repr(transparent)` delegates all representation concerns to
another type, so adding more representation hints is contradictory. Remove
either the `transparent` hint or the other hints, like this:

```
#[repr(transparent)]
struct Grams(f32);
```

Alternatively, move the other attributes to the contained type:

```
#[repr(C)]
struct Foo {
    x: i32,
    // ...
}

#[repr(transparent)]
struct FooWrapper(Foo);
```

Note that introducing another `struct` just to have a place for the other
attributes may have unintended side effects on the representation:

```
#[repr(transparent)]
struct Grams(f32);

#[repr(C)]
struct Float(f32);

#[repr(transparent)]
struct Grams2(Float); // this is not equivalent to `Grams` above
```

Here, `Grams2` is a not equivalent to `Grams` -- the former transparently wraps
a (non-transparent) struct containing a single float, while `Grams` is a
transparent wrapper around a float. This can make a difference for the ABI.
"##,

E0695: r##"
A `break` statement without a label appeared inside a labeled block.

Example of erroneous code:

```compile_fail,E0695
# #![feature(label_break_value)]
loop {
    'a: {
        break;
    }
}
```

Make sure to always label the `break`:

```
# #![feature(label_break_value)]
'l: loop {
    'a: {
        break 'l;
    }
}
```

Or if you want to `break` the labeled block:

```
# #![feature(label_break_value)]
loop {
    'a: {
        break 'a;
    }
    break;
}
```
"##,

E0697: r##"
A closure has been used as `static`.

Erroneous code example:

```compile_fail,E0697
fn main() {
    static || {}; // used as `static`
}
```

Closures cannot be used as `static`. They "save" the environment,
and as such a static closure would save only a static environment
which would consist only of variables with a static lifetime. Given
this it would be better to use a proper function. The easiest fix
is to remove the `static` keyword.
"##,

E0698: r##"
When using generators (or async) all type variables must be bound so a
generator can be constructed.

Erroneous code example:

```edition2018,compile-fail,E0698
async fn bar<T>() -> () {}

async fn foo() {
    bar().await; // error: cannot infer type for `T`
}
```

In the above example `T` is unknowable by the compiler.
To fix this you must bind `T` to a concrete type such as `String`
so that a generator can then be constructed:

```edition2018
async fn bar<T>() -> () {}

async fn foo() {
    bar::<String>().await;
    //   ^^^^^^^^ specify type explicitly
}
```
"##,

E0699: r##"
A method was called on a raw pointer whose inner type wasn't completely known.

For example, you may have done something like:

```compile_fail
# #![deny(warnings)]
let foo = &1;
let bar = foo as *const _;
if bar.is_null() {
    // ...
}
```

Here, the type of `bar` isn't known; it could be a pointer to anything. Instead,
specify a type for the pointer (preferably something that makes sense for the
thing you're pointing to):

```
let foo = &1;
let bar = foo as *const i32;
if bar.is_null() {
    // ...
}
```

Even though `is_null()` exists as a method on any raw pointer, Rust shows this
error because  Rust allows for `self` to have arbitrary types (behind the
arbitrary_self_types feature flag).

This means that someone can specify such a function:

```ignore (cannot-doctest-feature-doesnt-exist-yet)
impl Foo {
    fn is_null(self: *const Self) -> bool {
        // do something else
    }
}
```

and now when you call `.is_null()` on a raw pointer to `Foo`, there's ambiguity.

Given that we don't know what type the pointer is, and there's potential
ambiguity for some types, we disallow calling methods on raw pointers when
the type is unknown.
"##,

E0700: r##"
The `impl Trait` return type captures lifetime parameters that do not
appear within the `impl Trait` itself.

Erroneous code example:

```compile-fail,E0700
use std::cell::Cell;

trait Trait<'a> { }

impl<'a, 'b> Trait<'b> for Cell<&'a u32> { }

fn foo<'x, 'y>(x: Cell<&'x u32>) -> impl Trait<'y>
where 'x: 'y
{
    x
}
```

Here, the function `foo` returns a value of type `Cell<&'x u32>`,
which references the lifetime `'x`. However, the return type is
declared as `impl Trait<'y>` -- this indicates that `foo` returns
"some type that implements `Trait<'y>`", but it also indicates that
the return type **only captures data referencing the lifetime `'y`**.
In this case, though, we are referencing data with lifetime `'x`, so
this function is in error.

To fix this, you must reference the lifetime `'x` from the return
type. For example, changing the return type to `impl Trait<'y> + 'x`
would work:

```
use std::cell::Cell;

trait Trait<'a> { }

impl<'a,'b> Trait<'b> for Cell<&'a u32> { }

fn foo<'x, 'y>(x: Cell<&'x u32>) -> impl Trait<'y> + 'x
where 'x: 'y
{
    x
}
```
"##,

E0701: r##"
This error indicates that a `#[non_exhaustive]` attribute was incorrectly placed
on something other than a struct or enum.

Examples of erroneous code:

```compile_fail,E0701
#[non_exhaustive]
trait Foo { }
```
"##,

E0704: r##"
This error indicates that a incorrect visibility restriction was specified.

Example of erroneous code:

```compile_fail,E0704
mod foo {
    pub(foo) struct Bar {
        x: i32
    }
}
```

To make struct `Bar` only visible in module `foo` the `in` keyword should be
used:
```
mod foo {
    pub(in crate::foo) struct Bar {
        x: i32
    }
}
# fn main() {}
```

For more information see the Rust Reference on [Visibility].

[Visibility]: https://doc.rust-lang.org/reference/visibility-and-privacy.html
"##,

E0705: r##"
A `#![feature]` attribute was declared for a feature that is stable in
the current edition, but not in all editions.

Erroneous code example:

```ignore (limited to a warning during 2018 edition development)
#![feature(rust_2018_preview)]
#![feature(test_2018_feature)] // error: the feature
                               // `test_2018_feature` is
                               // included in the Rust 2018 edition
```
"##,

E0712: r##"
This error occurs because a borrow of a thread-local variable was made inside a
function which outlived the lifetime of the function.

Erroneous code example:

```compile_fail,E0712
#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn main() {
    let a = &FOO; // error: thread-local variable borrowed past end of function

    std::thread::spawn(move || {
        println!("{}", a);
    });
}
```
"##,

E0713: r##"
This error occurs when an attempt is made to borrow state past the end of the
lifetime of a type that implements the `Drop` trait.

Erroneous code example:

```compile_fail,E0713
#![feature(nll)]

pub struct S<'a> { data: &'a mut String }

impl<'a> Drop for S<'a> {
    fn drop(&mut self) { self.data.push_str("being dropped"); }
}

fn demo<'a>(s: S<'a>) -> &'a mut String { let p = &mut *s.data; p }
```

Here, `demo` tries to borrow the string data held within its
argument `s` and then return that borrow. However, `S` is
declared as implementing `Drop`.

Structs implementing the `Drop` trait have an implicit destructor that
gets called when they go out of scope. This destructor gets exclusive
access to the fields of the struct when it runs.

This means that when `s` reaches the end of `demo`, its destructor
gets exclusive access to its `&mut`-borrowed string data.  allowing
another borrow of that string data (`p`), to exist across the drop of
`s` would be a violation of the principle that `&mut`-borrows have
exclusive, unaliased access to their referenced data.

This error can be fixed by changing `demo` so that the destructor does
not run while the string-data is borrowed; for example by taking `S`
by reference:

```
pub struct S<'a> { data: &'a mut String }

impl<'a> Drop for S<'a> {
    fn drop(&mut self) { self.data.push_str("being dropped"); }
}

fn demo<'a>(s: &'a mut S<'a>) -> &'a mut String { let p = &mut *(*s).data; p }
```

Note that this approach needs a reference to S with lifetime `'a`.
Nothing shorter than `'a` will suffice: a shorter lifetime would imply
that after `demo` finishes executing, something else (such as the
destructor!) could access `s.data` after the end of that shorter
lifetime, which would again violate the `&mut`-borrow's exclusive
access.
"##,

E0714: r##"
A `#[marker]` trait contained an associated item.

The items of marker traits cannot be overridden, so there's no need to have them
when they cannot be changed per-type anyway.  If you wanted them for ergonomic
reasons, consider making an extension trait instead.
"##,

E0715: r##"
An `impl` for a `#[marker]` trait tried to override an associated item.

Because marker traits are allowed to have multiple implementations for the same
type, it's not allowed to override anything in those implementations, as it
would be ambiguous which override should actually be used.
"##,

E0716: r##"
This error indicates that a temporary value is being dropped
while a borrow is still in active use.

Erroneous code example:

```compile_fail,E0716
fn foo() -> i32 { 22 }
fn bar(x: &i32) -> &i32 { x }
let p = bar(&foo());
         // ------ creates a temporary
let q = *p;
```

Here, the expression `&foo()` is borrowing the expression
`foo()`. As `foo()` is a call to a function, and not the name of
a variable, this creates a **temporary** -- that temporary stores
the return value from `foo()` so that it can be borrowed.
You could imagine that `let p = bar(&foo());` is equivalent
to this:

```compile_fail,E0597
# fn foo() -> i32 { 22 }
# fn bar(x: &i32) -> &i32 { x }
let p = {
  let tmp = foo(); // the temporary
  bar(&tmp)
}; // <-- tmp is freed as we exit this block
let q = p;
```

Whenever a temporary is created, it is automatically dropped (freed)
according to fixed rules. Ordinarily, the temporary is dropped
at the end of the enclosing statement -- in this case, after the `let`.
This is illustrated in the example above by showing that `tmp` would
be freed as we exit the block.

To fix this problem, you need to create a local variable
to store the value in rather than relying on a temporary.
For example, you might change the original program to
the following:

```
fn foo() -> i32 { 22 }
fn bar(x: &i32) -> &i32 { x }
let value = foo(); // dropped at the end of the enclosing block
let p = bar(&value);
let q = *p;
```

By introducing the explicit `let value`, we allocate storage
that will last until the end of the enclosing block (when `value`
goes out of scope). When we borrow `&value`, we are borrowing a
local variable that already exists, and hence no temporary is created.

Temporaries are not always dropped at the end of the enclosing
statement. In simple cases where the `&` expression is immediately
stored into a variable, the compiler will automatically extend
the lifetime of the temporary until the end of the enclosing
block. Therefore, an alternative way to fix the original
program is to write `let tmp = &foo()` and not `let tmp = foo()`:

```
fn foo() -> i32 { 22 }
fn bar(x: &i32) -> &i32 { x }
let value = &foo();
let p = bar(value);
let q = *p;
```

Here, we are still borrowing `foo()`, but as the borrow is assigned
directly into a variable, the temporary will not be dropped until
the end of the enclosing block. Similar rules apply when temporaries
are stored into aggregate structures like a tuple or struct:

```
// Here, two temporaries are created, but
// as they are stored directly into `value`,
// they are not dropped until the end of the
// enclosing block.
fn foo() -> i32 { 22 }
let value = (&foo(), &foo());
```
"##,

E0718: r##"
This error indicates that a `#[lang = ".."]` attribute was placed
on the wrong type of item.

Examples of erroneous code:

```compile_fail,E0718
#![feature(lang_items)]

#[lang = "arc"]
static X: u32 = 42;
```
"##,

E0720: r##"
An `impl Trait` type expands to a recursive type.

An `impl Trait` type must be expandable to a concrete type that contains no
`impl Trait` types. For example the following example tries to create an
`impl Trait` type `T` that is equal to `[T, T]`:

```compile_fail,E0720
fn make_recursive_type() -> impl Sized {
    [make_recursive_type(), make_recursive_type()]
}
```
"##,

E0723: r##"
An feature unstable in `const` contexts was used.

Erroneous code example:

```compile_fail,E0723
trait T {}

impl T for () {}

const fn foo() -> impl T { // error: `impl Trait` in const fn is unstable
    ()
}
```

To enable this feature on a nightly version of rustc, add the `const_fn`
feature flag:

```
#![feature(const_fn)]

trait T {}

impl T for () {}

const fn foo() -> impl T {
    ()
}
```
"##,

E0725: r##"
A feature attribute named a feature that was disallowed in the compiler
command line flags.

Erroneous code example:

```ignore (can't specify compiler flags from doctests)
#![feature(never_type)] // error: the feature `never_type` is not in
                        // the list of allowed features
```

Delete the offending feature attribute, or add it to the list of allowed
features in the `-Z allow_features` flag.
"##,

E0728: r##"
[`await`] has been used outside [`async`] function or block.

Erroneous code examples:

```edition2018,compile_fail,E0728
# use std::pin::Pin;
# use std::future::Future;
# use std::task::{Context, Poll};
#
# struct WakeOnceThenComplete(bool);
#
# fn wake_and_yield_once() -> WakeOnceThenComplete {
#     WakeOnceThenComplete(false)
# }
#
# impl Future for WakeOnceThenComplete {
#     type Output = ();
#     fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
#         if self.0 {
#             Poll::Ready(())
#         } else {
#             cx.waker().wake_by_ref();
#             self.0 = true;
#             Poll::Pending
#         }
#     }
# }
#
fn foo() {
    wake_and_yield_once().await // `await` is used outside `async` context
}
```

[`await`] is used to suspend the current computation until the given
future is ready to produce a value. So it is legal only within
an [`async`] context, like an `async fn` or an `async` block.

```edition2018
# use std::pin::Pin;
# use std::future::Future;
# use std::task::{Context, Poll};
#
# struct WakeOnceThenComplete(bool);
#
# fn wake_and_yield_once() -> WakeOnceThenComplete {
#     WakeOnceThenComplete(false)
# }
#
# impl Future for WakeOnceThenComplete {
#     type Output = ();
#     fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
#         if self.0 {
#             Poll::Ready(())
#         } else {
#             cx.waker().wake_by_ref();
#             self.0 = true;
#             Poll::Pending
#         }
#     }
# }
#
async fn foo() {
    wake_and_yield_once().await // `await` is used within `async` function
}

fn bar(x: u8) -> impl Future<Output = u8> {
    async move {
        wake_and_yield_once().await; // `await` is used within `async` block
        x
    }
}
```

[`async`]: https://doc.rust-lang.org/std/keyword.async.html
[`await`]: https://doc.rust-lang.org/std/keyword.await.html
"##,

E0729: r##"
Support for Non-Lexical Lifetimes (NLL) has been included in the Rust compiler
since 1.31, and has been enabled on the 2015 edition since 1.36. The new borrow
checker for NLL uncovered some bugs in the old borrow checker, which in some
cases allowed unsound code to compile, resulting in memory safety issues.

### What do I do?

Change your code so the warning does no longer trigger. For backwards
compatibility, this unsound code may still compile (with a warning) right now.
However, at some point in the future, the compiler will no longer accept this
code and will throw a hard error.

### Shouldn't you fix the old borrow checker?

The old borrow checker has known soundness issues that are basically impossible
to fix. The new NLL-based borrow checker is the fix.

### Can I turn these warnings into errors by denying a lint?

No.

### When are these warnings going to turn into errors?

No formal timeline for turning the warnings into errors has been set. See
[GitHub issue 58781](https://github.com/rust-lang/rust/issues/58781) for more
information.

### Why do I get this message with code that doesn't involve borrowing?

There are some known bugs that trigger this message.
"##,

E0730: r##"
An array without a fixed length was pattern-matched.

Example of erroneous code:

```compile_fail,E0730
#![feature(const_generics)]

fn is_123<const N: usize>(x: [u32; N]) -> bool {
    match x {
        [1, 2, 3] => true, // error: cannot pattern-match on an
                           //        array without a fixed length
        _ => false
    }
}
```

Ensure that the pattern is consistent with the size of the matched
array. Additional elements can be matched with `..`:

```
#![feature(slice_patterns)]

let r = &[1, 2, 3, 4];
match r {
    &[a, b, ..] => { // ok!
        println!("a={}, b={}", a, b);
    }
}
```
"##,

E0731: r##"
An enum with the representation hint `repr(transparent)` had zero or more than
one variants.

Erroneous code example:

```compile_fail,E0731
#[repr(transparent)]
enum Status { // error: transparent enum needs exactly one variant, but has 2
    Errno(u32),
    Ok,
}
```

Because transparent enums are represented exactly like one of their variants at
run time, said variant must be uniquely determined. If there is no variant, or
if there are multiple variants, it is not clear how the enum should be
represented.
"##,

E0732: r##"
An `enum` with a discriminant must specify a `#[repr(inttype)]`.

A `#[repr(inttype)]` must be provided on an `enum` if it has a non-unit
variant with a discriminant, or where there are both unit variants with
discriminants and non-unit variants. This restriction ensures that there
is a well-defined way to extract a variant's discriminant from a value;
for instance:

```
#![feature(arbitrary_enum_discriminant)]

#[repr(u8)]
enum Enum {
    Unit = 3,
    Tuple(u16) = 2,
    Struct {
        a: u8,
        b: u16,
    } = 1,
}

fn discriminant(v : &Enum) -> u8 {
    unsafe { *(v as *const Enum as *const u8) }
}

assert_eq!(3, discriminant(&Enum::Unit));
assert_eq!(2, discriminant(&Enum::Tuple(5)));
assert_eq!(1, discriminant(&Enum::Struct{a: 7, b: 11}));
```
"##,

E0733: r##"
Recursion in an `async fn` requires boxing. For example, this will not compile:

```edition2018,compile_fail,E0733
async fn foo(n: usize) {
    if n > 0 {
        foo(n - 1).await;
    }
}
```

To achieve async recursion, the `async fn` needs to be desugared
such that the `Future` is explicit in the return type:

```edition2018,compile_fail,E0720
use std::future::Future;
fn foo_desugared(n: usize) -> impl Future<Output = ()> {
    async move {
        if n > 0 {
            foo_desugared(n - 1).await;
        }
    }
}
```

Finally, the future is wrapped in a pinned box:

```edition2018
use std::future::Future;
use std::pin::Pin;
fn foo_recursive(n: usize) -> Pin<Box<dyn Future<Output = ()>>> {
    Box::pin(async move {
        if n > 0 {
            foo_recursive(n - 1).await;
        }
    })
}
```

The `Box<...>` ensures that the result is of known size,
and the pin is required to keep it in the same place in memory.
"##,

E0734: r##"
A stability attribute has been used outside of the standard library.

Erroneous code examples:

```compile_fail,E0734
#[rustc_deprecated(since = "b", reason = "text")] // invalid
#[stable(feature = "a", since = "b")] // invalid
#[unstable(feature = "b", issue = "0")] // invalid
fn foo(){}
```

These attributes are meant to only be used by the standard library and are
rejected in your own crates.
"##,

E0735: r##"
Type parameter defaults cannot use `Self` on structs, enums, or unions.

Erroneous code example:

```compile_fail,E0735
struct Foo<X = Box<Self>> {
    field1: Option<X>,
    field2: Option<X>,
}
// error: type parameters cannot use `Self` in their defaults.
```
"##,

E0736: r##"
`#[track_caller]` and `#[naked]` cannot both be applied to the same function.

Erroneous code example:

```compile_fail,E0736
#![feature(track_caller)]

#[naked]
#[track_caller]
fn foo() {}
```

This is primarily due to ABI incompatibilities between the two attributes.
See [RFC 2091] for details on this and other limitations.

[RFC 2091]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md
"##,

E0737: r##"
`#[track_caller]` requires functions to have the `"Rust"` ABI for implicitly
receiving caller location. See [RFC 2091] for details on this and other
restrictions.

Erroneous code example:

```compile_fail,E0737
#![feature(track_caller)]

#[track_caller]
extern "C" fn foo() {}
```

[RFC 2091]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md
"##,

E0738: r##"
`#[track_caller]` cannot be used in traits yet. This is due to limitations in
the compiler which are likely to be temporary. See [RFC 2091] for details on
this and other restrictions.

Erroneous example with a trait method implementation:

```compile_fail,E0738
#![feature(track_caller)]

trait Foo {
    fn bar(&self);
}

impl Foo for u64 {
    #[track_caller]
    fn bar(&self) {}
}
```

Erroneous example with a blanket trait method implementation:

```compile_fail,E0738
#![feature(track_caller)]

trait Foo {
    #[track_caller]
    fn bar(&self) {}
    fn baz(&self);
}
```

Erroneous example with a trait method declaration:

```compile_fail,E0738
#![feature(track_caller)]

trait Foo {
    fn bar(&self) {}

    #[track_caller]
    fn baz(&self);
}
```

Note that while the compiler may be able to support the attribute in traits in
the future, [RFC 2091] prohibits their implementation without a follow-up RFC.

[RFC 2091]: https://github.com/rust-lang/rfcs/blob/master/text/2091-inline-semantic.md
"##,

E0740: r##"
A `union` cannot have fields with destructors.
"##,

E0741: r##"
Only `structural_match` types (that is, types that derive `PartialEq` and `Eq`)
may be used as the types of const generic parameters.

```compile_fail,E0741
#![feature(const_generics)]

struct A;

struct B<const X: A>; // error!
```

To fix this example, we derive `PartialEq` and `Eq`.

```
#![feature(const_generics)]

#[derive(PartialEq, Eq)]
struct A;

struct B<const X: A>; // ok!
```
"##,

E0742: r##"
Visibility is restricted to a module which isn't an ancestor of the current
item.

Erroneous code example:

```compile_fail,E0742,edition2018
pub mod Sea {}

pub (in crate::Sea) struct Shark; // error!

fn main() {}
```

To fix this error, we need to move the `Shark` struct inside the `Sea` module:

```edition2018
pub mod Sea {
    pub (in crate::Sea) struct Shark; // ok!
}

fn main() {}
```

Of course, you can do it as long as the module you're referring to is an
ancestor:

```edition2018
pub mod Earth {
    pub mod Sea {
        pub (in crate::Earth) struct Shark; // ok!
    }
}

fn main() {}
```
"##,

E0743: r##"
C-variadic has been used on a non-foreign function.

Erroneous code example:

```compile_fail,E0743
fn foo2(x: u8, ...) {} // error!
```

Only foreign functions can use C-variadic (`...`). It is used to give an
undefined number of parameters to a given function (like `printf` in C). The
equivalent in Rust would be to use macros directly.
"##,

;
//  E0006, // merged with E0005
//  E0008, // cannot bind by-move into a pattern guard
//  E0035, merged into E0087/E0089
//  E0036, merged into E0087/E0089
//  E0068,
//  E0085,
//  E0086,
//  E0101, // replaced with E0282
//  E0102, // replaced with E0282
//  E0103,
//  E0104,
//  E0122, // bounds in type aliases are ignored, turned into proper lint
//  E0123,
//  E0127,
//  E0129,
//  E0134,
//  E0135,
//  E0141,
//  E0153, unused error code
//  E0157, unused error code
//  E0159, // use of trait `{}` as struct constructor
//  E0163, // merged into E0071
//  E0167,
//  E0168,
//  E0172, // non-trait found in a type sum, moved to resolve
//  E0173, // manual implementations of unboxed closure traits are experimental
//  E0174,
//  E0182, // merged into E0229
    E0183,
//  E0187, // cannot infer the kind of the closure
//  E0188, // can not cast an immutable reference to a mutable pointer
//  E0189, // deprecated: can only cast a boxed pointer to a boxed object
//  E0190, // deprecated: can only cast a &-pointer to an &-object
//  E0194, // merged into E0403
//  E0196, // cannot determine a type for this closure
    E0203, // type parameter has more than one relaxed default bound,
           // and only one is supported
    E0208,
//  E0209, // builtin traits can only be implemented on structs or enums
    E0212, // cannot extract an associated type from a higher-ranked trait bound
//  E0213, // associated types are not accepted in this context
//  E0215, // angle-bracket notation is not stable with `Fn`
//  E0216, // parenthetical notation is only stable with `Fn`
//  E0217, // ambiguous associated type, defined in multiple supertraits
//  E0218, // no associated type defined
//  E0219, // associated type defined in higher-ranked supertrait
//  E0222, // Error code E0045 (variadic function must have C or cdecl calling
           // convention) duplicate
    E0224, // at least one non-builtin train is required for an object type
    E0226, // only a single explicit lifetime bound is permitted
    E0227, // ambiguous lifetime bound, explicit lifetime bound required
    E0228, // explicit lifetime bound required
//  E0233,
//  E0234,
//  E0235, // structure constructor specifies a structure of type but
//  E0236, // no lang item for range syntax
//  E0237, // no lang item for range syntax
//  E0238, // parenthesized parameters may only be used with a trait
//  E0239, // `next` method of `Iterator` trait has unexpected type
//  E0240,
//  E0241,
//  E0242,
//  E0245, // not a trait
//  E0246, // invalid recursive type
//  E0247,
//  E0248, // value used as a type, now reported earlier during resolution
           // as E0412
//  E0249,
//  E0257,
//  E0258,
//  E0272, // on_unimplemented #0
//  E0273, // on_unimplemented #1
//  E0274, // on_unimplemented #2
//  E0278, // requirement is not satisfied
    E0279, // requirement is not satisfied
    E0280, // requirement is not satisfied
//  E0285, // overflow evaluation builtin bounds
//  E0296, // replaced with a generic attribute input check
//  E0298, // cannot compare constants
//  E0299, // mismatched types between arms
//  E0300, // unexpanded macro
//  E0304, // expected signed integer constant
//  E0305, // expected constant
    E0311, // thing may not live long enough
    E0313, // lifetime of borrowed pointer outlives lifetime of captured
           // variable
    E0314, // closure outlives stack frame
    E0315, // cannot invoke closure outside of its lifetime
    E0316, // nested quantification of lifetimes
//  E0319, // trait impls for defaulted traits allowed just for structs/enums
    E0320, // recursive overflow during dropck
//  E0372, // coherence not object safe
    E0377, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with the same definition
//  E0385, // {} in an aliasable location
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
//  E0427, merged into 530
    E0456, // plugin `..` is not available for triple `..`
    E0457, // plugin `..` only found in rlib format, but must be available...
    E0460, // found possibly newer version of crate `..`
    E0461, // couldn't find crate `..` with expected target triple ..
    E0462, // found staticlib `..` instead of rlib or dylib
    E0464, // multiple matching crates for `..`
    E0465, // multiple .. candidates for `..` found
//  E0467, removed
//  E0470, removed
//  E0471, // constant evaluation error (in pattern)
    E0472, // asm! is unsupported on this target
    E0473, // dereference of reference outside its lifetime
    E0474, // captured variable `..` does not outlive the enclosing closure
    E0475, // index of slice outside its lifetime
    E0476, // lifetime of the source pointer does not outlive lifetime bound...
    E0477, // the type `..` does not fulfill the required lifetime...
    E0479, // the type `..` (provided as the value of a type parameter) is...
    E0480, // lifetime of method receiver does not outlive the method call
    E0481, // lifetime of function argument does not outlive the function call
    E0482, // lifetime of return value does not outlive the function call
    E0483, // lifetime of operand does not outlive the operation
    E0484, // reference is not valid at the time of borrow
    E0485, // automatically reference is not valid at the time of borrow
    E0486, // type of expression contains references that are not valid during..
    E0487, // unsafe use of destructor: destructor might be called while...
    E0488, // lifetime of variable does not enclose its declaration
    E0489, // type/lifetime parameter not in scope here
    E0490, // a value of type `..` is borrowed for too long
    E0498,  // malformed plugin attribute
    E0514, // metadata version mismatch
    E0519, // local crate and dependency have same (crate-name, disambiguator)
    // two dependencies have same (crate-name, disambiguator) but different SVH
    E0521, // borrowed data escapes outside of closure
    E0523,
//  E0526, // shuffle indices are not constant
    E0539, // incorrect meta item
    E0540, // multiple rustc_deprecated attributes
    E0542, // missing 'since'
    E0543, // missing 'reason'
    E0544, // multiple stability levels
    E0545, // incorrect 'issue'
    E0546, // missing 'feature'
    E0547, // missing 'issue'
//  E0548, // replaced with a generic attribute input check
    // rustc_deprecated attribute must be paired with either stable or unstable
    // attribute
    E0549,
    E0553, // multiple rustc_const_unstable attributes
//  E0555, // replaced with a generic attribute input check
//  E0558, // replaced with a generic attribute input check
//  E0563, // cannot determine a type for this `impl Trait` removed in 6383de15
//  E0564, // only named lifetimes are allowed in `impl Trait`,
           // but `{}` was found in the type `{}`
    E0594, // cannot assign to {}
//  E0598, // lifetime of {} is too short to guarantee its contents can be...
//  E0611, // merged into E0616
//  E0612, // merged into E0609
//  E0613, // Removed (merged with E0609)
    E0623, // lifetime mismatch where both parameters are anonymous regions
    E0625, // thread-local statics cannot be accessed at compile-time
    E0627, // yield statement outside of generator literal
    E0628, // generators cannot have explicit parameters
    E0629, // missing 'feature' (rustc_const_unstable)
    // rustc_const_unstable attribute must be paired with stable/unstable
    // attribute
    E0630,
    E0631, // type mismatch in closure arguments
    E0632, // cannot provide explicit generic arguments when `impl Trait` is
           // used in argument position
    E0634, // type has conflicting packed representaton hints
    E0637, // "'_" is not a valid lifetime bound
    E0640, // infer outlives requirements
    E0641, // cannot cast to/from a pointer with an unknown kind
//  E0645, // trait aliases not finished
    E0657, // `impl Trait` can only capture lifetimes bound at the fn level
    E0667, // `impl Trait` in projections
    E0687, // in-band lifetimes cannot be used in `fn`/`Fn` syntax
    E0688, // in-band lifetimes cannot be mixed with explicit lifetime binders
    E0693, // incorrect `repr(align)` attribute format
//  E0694, // an unknown tool name found in scoped attributes
    E0696, // `continue` pointing to a labeled block
//  E0702, // replaced with a generic attribute input check
    E0703, // invalid ABI
    E0706, // `async fn` in trait
//  E0707, // multiple elided lifetimes used in arguments of `async fn`
    E0708, // `async` non-`move` closures with parameters are not currently
           // supported
//  E0709, // multiple different lifetimes used in arguments of `async fn`
    E0710, // an unknown tool name found in scoped lint
    E0711, // a feature has been declared with conflicting stability attributes
    E0717, // rustc_promotable without stability attribute
    E0719, // duplicate values for associated type binding
//  E0721, // `await` keyword
    E0722, // Malformed `#[optimize]` attribute
    E0724, // `#[ffi_returns_twice]` is only allowed in foreign functions
    E0726, // non-explicit (not `'_`) elided lifetime in unsupported position
    E0727, // `async` generators are not yet supported
    E0739, // invalid track_caller application/syntax
}
