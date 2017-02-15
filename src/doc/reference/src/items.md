# Items

An _item_ is a component of a crate. Items are organized within a crate by a
nested set of [modules](#modules). Every crate has a single "outermost"
anonymous module; all further items within the crate have [paths](#paths)
within the module tree of the crate.

Items are entirely determined at compile-time, generally remain fixed during
execution, and may reside in read-only memory.

There are several kinds of item:

* [`extern crate` declarations](#extern-crate-declarations)
* [`use` declarations](#use-declarations)
* [modules](#modules)
* [function definitions](#functions)
* [`extern` blocks](#external-blocks)
* [type definitions](grammar.html#type-definitions)
* [struct definitions](#structs)
* [enumeration definitions](#enumerations)
* [constant items](#constant-items)
* [static items](#static-items)
* [trait definitions](#traits)
* [implementations](#implementations)

Some items form an implicit scope for the declaration of sub-items. In other
words, within a function or module, declarations of items can (in many cases)
be mixed with the statements, control blocks, and similar artifacts that
otherwise compose the item body. The meaning of these scoped items is the same
as if the item was declared outside the scope &mdash; it is still a static item
&mdash; except that the item's *path name* within the module namespace is
qualified by the name of the enclosing item, or is private to the enclosing
item (in the case of functions). The grammar specifies the exact locations in
which sub-item declarations may appear.

## Type Parameters

All items except modules, constants and statics may be *parameterized* by type.
Type parameters are given as a comma-separated list of identifiers enclosed in
angle brackets (`<...>`), after the name of the item and before its definition.
The type parameters of an item are considered "part of the name", not part of
the type of the item. A referencing [path](#paths) must (in principle) provide
type arguments as a list of comma-separated types enclosed within angle
brackets, in order to refer to the type-parameterized item. In practice, the
type-inference system can usually infer such argument types from context. There
are no general type-parametric types, only type-parametric items. That is, Rust
has no notion of type abstraction: there are no higher-ranked (or "forall") types
abstracted over other types, though higher-ranked types do exist for lifetimes.

## Modules

A module is a container for zero or more [items](#items).

A _module item_ is a module, surrounded in braces, named, and prefixed with the
keyword `mod`. A module item introduces a new, named module into the tree of
modules making up a crate. Modules can nest arbitrarily.

An example of a module:

```
mod math {
    type Complex = (f64, f64);
    fn sin(f: f64) -> f64 {
        /* ... */
# panic!();
    }
    fn cos(f: f64) -> f64 {
        /* ... */
# panic!();
    }
    fn tan(f: f64) -> f64 {
        /* ... */
# panic!();
    }
}
```

Modules and types share the same namespace. Declaring a named type with
the same name as a module in scope is forbidden: that is, a type definition,
trait, struct, enumeration, or type parameter can't shadow the name of a module
in scope, or vice versa.

A module without a body is loaded from an external file, by default with the
same name as the module, plus the `.rs` extension. When a nested submodule is
loaded from an external file, it is loaded from a subdirectory path that
mirrors the module hierarchy.

```{.ignore}
// Load the `vec` module from `vec.rs`
mod vec;

mod thread {
    // Load the `local_data` module from `thread/local_data.rs`
    // or `thread/local_data/mod.rs`.
    mod local_data;
}
```

The directories and files used for loading external file modules can be
influenced with the `path` attribute.

```{.ignore}
#[path = "thread_files"]
mod thread {
    // Load the `local_data` module from `thread_files/tls.rs`
    #[path = "tls.rs"]
    mod local_data;
}
```

### Extern crate declarations

An _`extern crate` declaration_ specifies a dependency on an external crate.
The external crate is then bound into the declaring scope as the `ident`
provided in the `extern_crate_decl`.

The external crate is resolved to a specific `soname` at compile time, and a
runtime linkage requirement to that `soname` is passed to the linker for
loading at runtime. The `soname` is resolved at compile time by scanning the
compiler's library path and matching the optional `crateid` provided against
the `crateid` attributes that were declared on the external crate when it was
compiled. If no `crateid` is provided, a default `name` attribute is assumed,
equal to the `ident` given in the `extern_crate_decl`.

Three examples of `extern crate` declarations:

```{.ignore}
extern crate pcre;

extern crate std; // equivalent to: extern crate std as std;

extern crate std as ruststd; // linking to 'std' under another name
```

When naming Rust crates, hyphens are disallowed. However, Cargo packages may
make use of them. In such case, when `Cargo.toml` doesn't specify a crate name,
Cargo will transparently replace `-` with `_` (Refer to [RFC 940] for more
details).

Here is an example:

```{.ignore}
// Importing the Cargo package hello-world
extern crate hello_world; // hyphen replaced with an underscore
```

[RFC 940]: https://github.com/rust-lang/rfcs/blob/master/text/0940-hyphens-considered-harmful.md

### Use declarations

A _use declaration_ creates one or more local name bindings synonymous with
some other [path](#paths). Usually a `use` declaration is used to shorten the
path required to refer to a module item. These declarations may appear in
[modules](#modules) and [blocks](grammar.html#block-expressions), usually at the top.

> **Note**: Unlike in many languages,
> `use` declarations in Rust do *not* declare linkage dependency with external crates.
> Rather, [`extern crate` declarations](#extern-crate-declarations) declare linkage dependencies.

Use declarations support a number of convenient shortcuts:

* Rebinding the target name as a new local name, using the syntax `use p::q::r as x;`
* Simultaneously binding a list of paths differing only in their final element,
  using the glob-like brace syntax `use a::b::{c,d,e,f};`
* Binding all paths matching a given prefix, using the asterisk wildcard syntax
  `use a::b::*;`
* Simultaneously binding a list of paths differing only in their final element
  and their immediate parent module, using the `self` keyword, such as
  `use a::b::{self, c, d};`

An example of `use` declarations:

```rust
use std::option::Option::{Some, None};
use std::collections::hash_map::{self, HashMap};

fn foo<T>(_: T){}
fn bar(map1: HashMap<String, usize>, map2: hash_map::HashMap<String, usize>){}

fn main() {
    // Equivalent to 'foo(vec![std::option::Option::Some(1.0f64),
    // std::option::Option::None]);'
    foo(vec![Some(1.0f64), None]);

    // Both `hash_map` and `HashMap` are in scope.
    let map1 = HashMap::new();
    let map2 = hash_map::HashMap::new();
    bar(map1, map2);
}
```

Like items, `use` declarations are private to the containing module, by
default. Also like items, a `use` declaration can be public, if qualified by
the `pub` keyword. Such a `use` declaration serves to _re-export_ a name. A
public `use` declaration can therefore _redirect_ some public name to a
different target definition: even a definition with a private canonical path,
inside a different module. If a sequence of such redirections form a cycle or
cannot be resolved unambiguously, they represent a compile-time error.

An example of re-exporting:

```
# fn main() { }
mod quux {
    pub use quux::foo::{bar, baz};

    pub mod foo {
        pub fn bar() { }
        pub fn baz() { }
    }
}
```

In this example, the module `quux` re-exports two public names defined in
`foo`.

Also note that the paths contained in `use` items are relative to the crate
root. So, in the previous example, the `use` refers to `quux::foo::{bar,
baz}`, and not simply to `foo::{bar, baz}`. This also means that top-level
module declarations should be at the crate root if direct usage of the declared
modules within `use` items is desired. It is also possible to use `self` and
`super` at the beginning of a `use` item to refer to the current and direct
parent modules respectively. All rules regarding accessing declared modules in
`use` declarations apply to both module declarations and `extern crate`
declarations.

An example of what will and will not work for `use` items:

```
# #![allow(unused_imports)]
use foo::baz::foobaz;    // good: foo is at the root of the crate

mod foo {

    mod example {
        pub mod iter {}
    }

    use foo::example::iter; // good: foo is at crate root
//  use example::iter;      // bad:  example is not at the crate root
    use self::baz::foobaz;  // good: self refers to module 'foo'
    use foo::bar::foobar;   // good: foo is at crate root

    pub mod bar {
        pub fn foobar() { }
    }

    pub mod baz {
        use super::bar::foobar; // good: super refers to module 'foo'
        pub fn foobaz() { }
    }
}

fn main() {}
```

## Functions

A _function item_ defines a sequence of [statements](#statements) and a
final [expression](#expressions), along with a name and a set of
parameters. Other than a name, all these are optional.
Functions are declared with the keyword `fn`. Functions may declare a
set of *input* [*variables*](#variables) as parameters, through which the caller
passes arguments into the function, and the *output* [*type*](#types)
of the value the function will return to its caller on completion.

A function may also be copied into a first-class *value*, in which case the
value has the corresponding [*function type*](#function-types), and can be used
otherwise exactly as a function item (with a minor additional cost of calling
the function indirectly).

Every control path in a function logically ends with a `return` expression or a
diverging expression. If the outermost block of a function has a
value-producing expression in its final-expression position, that expression is
interpreted as an implicit `return` expression applied to the final-expression.

An example of a function:

```
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

As with `let` bindings, function arguments are irrefutable patterns, so any
pattern that is valid in a let binding is also valid as an argument.

```
fn first((value, _): (i32, i32)) -> i32 { value }
```


### Generic functions

A _generic function_ allows one or more _parameterized types_ to appear in its
signature. Each type parameter must be explicitly declared in an
angle-bracket-enclosed and comma-separated list, following the function name.

```rust,ignore
// foo is generic over A and B

fn foo<A, B>(x: A, y: B) {
```

Inside the function signature and body, the name of the type parameter can be
used as a type name. [Trait](#traits) bounds can be specified for type parameters
to allow methods with that trait to be called on values of that type. This is
specified using the `where` syntax:

```rust,ignore
fn foo<T>(x: T) where T: Debug {
```

When a generic function is referenced, its type is instantiated based on the
context of the reference. For example, calling the `foo` function here:

```
use std::fmt::Debug;

fn foo<T>(x: &[T]) where T: Debug {
    // details elided
    # ()
}

foo(&[1, 2]);
```

will instantiate type parameter `T` with `i32`.

The type parameters can also be explicitly supplied in a trailing
[path](#paths) component after the function name. This might be necessary if
there is not sufficient context to determine the type parameters. For example,
`mem::size_of::<u32>() == 4`.

### Diverging functions

A special kind of function can be declared with a `!` character where the
output type would normally be. For example:

```
fn my_err(s: &str) -> ! {
    println!("{}", s);
    panic!();
}
```

We call such functions "diverging" because they never return a value to the
caller. Every control path in a diverging function must end with a `panic!()` or
a call to another diverging function on every control path. The `!` annotation
does *not* denote a type.

It might be necessary to declare a diverging function because as mentioned
previously, the typechecker checks that every control path in a function ends
with a [`return`](#return-expressions) or diverging expression. So, if `my_err`
were declared without the `!` annotation, the following code would not
typecheck:

```
# fn my_err(s: &str) -> ! { panic!() }

fn f(i: i32) -> i32 {
    if i == 42 {
        return 42;
    }
    else {
        my_err("Bad number!");
    }
}
```

This will not compile without the `!` annotation on `my_err`, since the `else`
branch of the conditional in `f` does not return an `i32`, as required by the
signature of `f`. Adding the `!` annotation to `my_err` informs the
typechecker that, should control ever enter `my_err`, no further type judgments
about `f` need to hold, since control will never resume in any context that
relies on those judgments. Thus the return type on `f` only needs to reflect
the `if` branch of the conditional.

### Extern functions

Extern functions are part of Rust's foreign function interface, providing the
opposite functionality to [external blocks](#external-blocks). Whereas
external blocks allow Rust code to call foreign code, extern functions with
bodies defined in Rust code _can be called by foreign code_. They are defined
in the same way as any other Rust function, except that they have the `extern`
modifier.

```
// Declares an extern fn, the ABI defaults to "C"
extern fn new_i32() -> i32 { 0 }

// Declares an extern fn with "stdcall" ABI
extern "stdcall" fn new_i32_stdcall() -> i32 { 0 }
```

Unlike normal functions, extern fns have type `extern "ABI" fn()`. This is the
same type as the functions declared in an extern block.

```
# extern fn new_i32() -> i32 { 0 }
let fptr: extern "C" fn() -> i32 = new_i32;
```

Extern functions may be called directly from Rust code as Rust uses large,
contiguous stack segments like C.

## Type aliases

A _type alias_ defines a new name for an existing [type](#types). Type
aliases are declared with the keyword `type`. Every value has a single,
specific type, but may implement several different traits, or be compatible with
several different type constraints.

For example, the following defines the type `Point` as a synonym for the type
`(u8, u8)`, the type of pairs of unsigned 8 bit integers:

```
type Point = (u8, u8);
let p: Point = (41, 68);
```

Currently a type alias to an enum type cannot be used to qualify the
constructors:

```
enum E { A }
type F = E;
let _: F = E::A;  // OK
// let _: F = F::A;  // Doesn't work
```

## Structs

A _struct_ is a nominal [struct type](#struct-types) defined with the
keyword `struct`.

An example of a `struct` item and its use:

```
struct Point {x: i32, y: i32}
let p = Point {x: 10, y: 11};
let px: i32 = p.x;
```

A _tuple struct_ is a nominal [tuple type](#tuple-types), also defined with
the keyword `struct`. For example:

```
struct Point(i32, i32);
let p = Point(10, 11);
let px: i32 = match p { Point(x, _) => x };
```

A _unit-like struct_ is a struct without any fields, defined by leaving off
the list of fields entirely. Such a struct implicitly defines a constant of
its type with the same name. For example:

```
struct Cookie;
let c = [Cookie, Cookie {}, Cookie, Cookie {}];
```

is equivalent to

```
struct Cookie {}
const Cookie: Cookie = Cookie {};
let c = [Cookie, Cookie {}, Cookie, Cookie {}];
```

The precise memory layout of a struct is not specified. One can specify a
particular layout using the [`repr` attribute](#ffi-attributes).

## Enumerations

An _enumeration_ is a simultaneous definition of a nominal [enumerated
type](#enumerated-types) as well as a set of *constructors*, that can be used
to create or pattern-match values of the corresponding enumerated type.

Enumerations are declared with the keyword `enum`.

An example of an `enum` item and its use:

```
enum Animal {
    Dog,
    Cat,
}

let mut a: Animal = Animal::Dog;
a = Animal::Cat;
```

Enumeration constructors can have either named or unnamed fields:

```rust
enum Animal {
    Dog (String, f64),
    Cat { name: String, weight: f64 },
}

let mut a: Animal = Animal::Dog("Cocoa".to_string(), 37.2);
a = Animal::Cat { name: "Spotty".to_string(), weight: 2.7 };
```

In this example, `Cat` is a _struct-like enum variant_,
whereas `Dog` is simply called an enum variant.

Each enum value has a _discriminant_ which is an integer associated to it. You
can specify it explicitly:

```
enum Foo {
    Bar = 123,
}
```

The right hand side of the specification is interpreted as an `isize` value,
but the compiler is allowed to use a smaller type in the actual memory layout.
The [`repr` attribute](#ffi-attributes) can be added in order to change
the type of the right hand side and specify the memory layout.

If a discriminant isn't specified, they start at zero, and add one for each
variant, in order.

You can cast an enum to get its discriminant:

```
# enum Foo { Bar = 123 }
let x = Foo::Bar as u32; // x is now 123u32
```

This only works as long as none of the variants have data attached. If
it were `Bar(i32)`, this is disallowed.

## Constant items

A *constant item* is a named _constant value_ which is not associated with a
specific memory location in the program. Constants are essentially inlined
wherever they are used, meaning that they are copied directly into the relevant
context when used. References to the same constant are not necessarily
guaranteed to refer to the same memory address.

Constant values must not have destructors, and otherwise permit most forms of
data. Constants may refer to the address of other constants, in which case the
address will have elided lifetimes where applicable, otherwise – in most cases –
defaulting to the `static` lifetime. (See below on [static lifetime elision].)
The compiler is, however, still at liberty to translate the constant many times,
so the address referred to may not be stable.

[static lifetime elision]: #static-lifetime-elision

Constants must be explicitly typed. The type may be `bool`, `char`, a number, or
a type derived from those primitive types. The derived types are references with
the `static` lifetime, fixed-size arrays, tuples, enum variants, and structs.

```rust
const BIT1: u32 = 1 << 0;
const BIT2: u32 = 1 << 1;

const BITS: [u32; 2] = [BIT1, BIT2];
const STRING: &'static str = "bitstring";

struct BitsNStrings<'a> {
    mybits: [u32; 2],
    mystring: &'a str,
}

const BITS_N_STRINGS: BitsNStrings<'static> = BitsNStrings {
    mybits: BITS,
    mystring: STRING,
};
```

## Static items

A *static item* is similar to a *constant*, except that it represents a precise
memory location in the program. A static is never "inlined" at the usage site,
and all references to it refer to the same memory location. Static items have
the `static` lifetime, which outlives all other lifetimes in a Rust program.
Static items may be placed in read-only memory if they do not contain any
interior mutability.

Statics may contain interior mutability through the `UnsafeCell` language item.
All access to a static is safe, but there are a number of restrictions on
statics:

* Statics may not contain any destructors.
* The types of static values must ascribe to `Sync` to allow thread-safe access.
* Statics may not refer to other statics by value, only by reference.
* Constants cannot refer to statics.

Constants should in general be preferred over statics, unless large amounts of
data are being stored, or single-address and mutability properties are required.

### Mutable statics

If a static item is declared with the `mut` keyword, then it is allowed to
be modified by the program. One of Rust's goals is to make concurrency bugs
hard to run into, and this is obviously a very large source of race conditions
or other bugs. For this reason, an `unsafe` block is required when either
reading or writing a mutable static variable. Care should be taken to ensure
that modifications to a mutable static are safe with respect to other threads
running in the same process.

Mutable statics are still very useful, however. They can be used with C
libraries and can also be bound from C libraries (in an `extern` block).

```rust
# fn atomic_add(_: &mut u32, _: u32) -> u32 { 2 }

static mut LEVELS: u32 = 0;

// This violates the idea of no shared state, and this doesn't internally
// protect against races, so this function is `unsafe`
unsafe fn bump_levels_unsafe1() -> u32 {
    let ret = LEVELS;
    LEVELS += 1;
    return ret;
}

// Assuming that we have an atomic_add function which returns the old value,
// this function is "safe" but the meaning of the return value may not be what
// callers expect, so it's still marked as `unsafe`
unsafe fn bump_levels_unsafe2() -> u32 {
    return atomic_add(&mut LEVELS, 1);
}
```

Mutable statics have the same restrictions as normal statics, except that the
type of the value is not required to ascribe to `Sync`.

### `'static` lifetime elision

[Unstable] Both constant and static declarations of reference types have
*implicit* `'static` lifetimes unless an explicit lifetime is specified. As
such, the constant declarations involving `'static` above may be written
without the lifetimes. Returning to our previous example:

```rust
# #![feature(static_in_const)]
const BIT1: u32 = 1 << 0;
const BIT2: u32 = 1 << 1;

const BITS: [u32; 2] = [BIT1, BIT2];
const STRING: &str = "bitstring";

struct BitsNStrings<'a> {
    mybits: [u32; 2],
    mystring: &'a str,
}

const BITS_N_STRINGS: BitsNStrings = BitsNStrings {
    mybits: BITS,
    mystring: STRING,
};
```

Note that if the `static` or `const` items include function or closure
references, which themselves include references, the compiler will first try the
standard elision rules ([see discussion in the nomicon][elision-nomicon]). If it
is unable to resolve the lifetimes by its usual rules, it will default to using
the `'static` lifetime. By way of example:

[elision-nomicon]: https://doc.rust-lang.org/nomicon/lifetime-elision.html

```rust,ignore
// Resolved as `fn<'a>(&'a str) -> &'a str`.
const RESOLVED_SINGLE: fn(&str) -> &str = ..

// Resolved as `Fn<'a, 'b, 'c>(&'a Foo, &'b Bar, &'c Baz) -> usize`.
const RESOLVED_MULTIPLE: Fn(&Foo, &Bar, &Baz) -> usize = ..

// There is insufficient information to bound the return reference lifetime
// relative to the argument lifetimes, so the signature is resolved as
// `Fn(&'static Foo, &'static Bar) -> &'static Baz`.
const RESOLVED_STATIC: Fn(&Foo, &Bar) -> &Baz = ..
```
