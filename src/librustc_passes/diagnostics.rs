// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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

E0010: r##"
The value of statics and constants must be known at compile time, and they live
for the entire lifetime of a program. Creating a boxed value allocates memory on
the heap at runtime, and therefore cannot be done at compile time. Erroneous
code example:

```compile_fail
#![feature(box_syntax)]

const CON : Box<i32> = box 0;
```
"##,

E0011: r##"
Initializers for constants and statics are evaluated at compile time.
User-defined operators rely on user-defined functions, which cannot be evaluated
at compile time.

Erroneous code example:

```compile_fail
use std::ops::Index;

struct Foo { a: u8 }

impl Index<u8> for Foo {
    type Output = u8;

    fn index<'a>(&'a self, idx: u8) -> &'a u8 { &self.a }
}

const a: Foo = Foo { a: 0u8 };
const b: u8 = a[0]; // Index trait is defined by the user, bad!
```

Only operators on builtin types are allowed.

Example:

```
const a: &'static [i32] = &[1, 2, 3];
const b: i32 = a[0]; // Ok!
```
"##,

E0013: r##"
Static and const variables can refer to other const variables. But a const
variable cannot refer to a static variable. For example, `Y` cannot refer to
`X` here:

```compile_fail
static X: i32 = 42;
const Y: i32 = X;
```

To fix this, the value can be extracted as a const and then used:

```
const A: i32 = 42;
static X: i32 = A;
const Y: i32 = A;
```
"##,

E0014: r##"
Constants can only be initialized by a constant value or, in a future
version of Rust, a call to a const function. This error indicates the use
of a path (like a::b, or x) denoting something other than one of these
allowed items. Erroneous code xample:

```compile_fail
const FOO: i32 = { let x = 0; x }; // 'x' isn't a constant nor a function!
```

To avoid it, you have to replace the non-constant value:

```
const FOO: i32 = { const X : i32 = 0; X };
// or even:
const FOO2: i32 = { 0 }; // but brackets are useless here
```
"##,

// FIXME(#24111) Change the language here when const fn stabilizes
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

E0016: r##"
Blocks in constants may only contain items (such as constant, function
definition, etc...) and a tail expression. Erroneous code example:

```compile_fail
const FOO: i32 = { let x = 0; x }; // 'x' isn't an item!
```

To avoid it, you have to replace the non-item object:

```
const FOO: i32 = { const X : i32 = 0; X };
```
"##,

E0017: r##"
References in statics and constants may only refer to immutable values.
Erroneous code example:

```compile_fail
static X: i32 = 1;
const C: i32 = 2;

// these three are not allowed:
const CR: &'static mut i32 = &mut C;
static STATIC_REF: &'static mut i32 = &mut X;
static CONST_REF: &'static mut i32 = &mut C;
```

Statics are shared everywhere, and if they refer to mutable data one might
violate memory safety since holding multiple mutable references to shared data
is not allowed.

If you really want global mutable state, try using `static mut` or a global
`UnsafeCell`.
"##,

E0018: r##"

The value of static and constant integers must be known at compile time. You
can't cast a pointer to an integer because the address of a pointer can
vary.

For example, if you write:

```compile_fail
static MY_STATIC: u32 = 42;
static MY_STATIC_ADDR: usize = &MY_STATIC as *const _ as usize;
static WHAT: usize = (MY_STATIC_ADDR^17) + MY_STATIC_ADDR;
```

Then `MY_STATIC_ADDR` would contain the address of `MY_STATIC`. However,
the address can change when the program is linked, as well as change
between different executions due to ASLR, and many linkers would
not be able to calculate the value of `WHAT`.

On the other hand, static and constant pointers can point either to
a known numeric address or to the address of a symbol.

```
static MY_STATIC_ADDR: &'static u32 = &MY_STATIC;
// ... and also
static MY_STATIC_ADDR2: *const u32 = &MY_STATIC;

const CONST_ADDR: *const u8 = 0x5f3759df as *const u8;
```

This does not pose a problem by itself because they can't be
accessed directly.
"##,

E0019: r##"
A function call isn't allowed in the const's initialization expression
because the expression's value must be known at compile-time. Erroneous code
example:

```compile_fail
enum Test {
    V1
}

impl Test {
    fn test(&self) -> i32 {
        12
    }
}

fn main() {
    const FOO: Test = Test::V1;

    const A: i32 = FOO.test(); // You can't call Test::func() here !
}
```

Remember: you can't use a function call inside a const's initialization
expression! However, you can totally use it anywhere else:

```
fn main() {
    const FOO: Test = Test::V1;

    FOO.func(); // here is good
    let x = FOO.func(); // or even here!
}
```
"##,

E0022: r##"
Constant functions are not allowed to mutate anything. Thus, binding to an
argument with a mutable pattern is not allowed. For example,

```compile_fail
const fn foo(mut x: u8) {
    // do stuff
}
```

Is incorrect because the function body may not mutate `x`.

Remove any mutable bindings from the argument list to fix this error. In case
you need to mutate the argument, try lazily initializing a global variable
instead of using a `const fn`, or refactoring the code to a functional style to
avoid mutation if possible.
"##,

E0030: r##"
When matching against a range, the compiler verifies that the range is
non-empty.  Range patterns include both end-points, so this is equivalent to
requiring the start of the range to be less than or equal to the end of the
range.

For example:

```compile_fail
match 5u32 {
    // This range is ok, albeit pointless.
    1 ... 1 => {}
    // This range is empty, and the compiler can tell.
    1000 ... 5 => {}
}
```
"##,

E0161: r##"
In Rust, you can only move a value when its size is known at compile time.

To work around this restriction, consider "hiding" the value behind a reference:
either `&x` or `&mut x`. Since a reference has a fixed size, this lets you move
it around as usual.
"##,

E0265: r##"
This error indicates that a static or constant references itself.
All statics and constants need to resolve to a value in an acyclic manner.

For example, neither of the following can be sensibly compiled:

```compile_fail
const X: u32 = X;
```

```compile_fail
const X: u32 = Y;
const Y: u32 = X;
```
"##,

E0267: r##"
This error indicates the use of a loop keyword (`break` or `continue`) inside a
closure but outside of any loop. Erroneous code example:

```compile_fail
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

```compile_fail
fn some_func() {
    break; // error: `break` outside of loop
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

E0378: r##"
Method calls that aren't calls to inherent `const` methods are disallowed
in statics, constants, and constant functions.

For example:

```compile_fail
const BAZ: i32 = Foo(25).bar(); // error, `bar` isn't `const`

struct Foo(i32);

impl Foo {
    const fn foo(&self) -> i32 {
        self.bar() // error, `bar` isn't `const`
    }

    fn bar(&self) -> i32 { self.0 }
}
```

For more information about `const fn`'s, see [RFC 911].

[RFC 911]: https://github.com/rust-lang/rfcs/blob/master/text/0911-const-fn.md
"##,

E0394: r##"
From [RFC 246]:

 > It is invalid for a static to reference another static by value. It is
 > required that all references be borrowed.

[RFC 246]: https://github.com/rust-lang/rfcs/pull/246
"##,


E0395: r##"
The value assigned to a constant scalar must be known at compile time,
which is not the case when comparing raw pointers.

Erroneous code example:

```compile_fail
static FOO: i32 = 42;
static BAR: i32 = 42;

static BAZ: bool = { (&FOO as *const i32) == (&BAR as *const i32) };
// error: raw pointers cannot be compared in statics!
```

The address assigned by the linker to `FOO` and `BAR` may or may not
be identical, so the value of `BAZ` can't be determined.

If you want to do the comparison, please do it at run-time.

For example:

```
static FOO: i32 = 42;
static BAR: i32 = 42;

let baz: bool = { (&FOO as *const i32) == (&BAR as *const i32) };
// baz isn't a constant expression so it's ok
```
"##,

E0396: r##"
The value behind a raw pointer can't be determined at compile-time
(or even link-time), which means it can't be used in a constant
expression. Erroneous code example:

```compile_fail
const REG_ADDR: *const u8 = 0x5f3759df as *const u8;

const VALUE: u8 = unsafe { *REG_ADDR };
// error: raw pointers cannot be dereferenced in constants
```

A possible fix is to dereference your pointer at some point in run-time.

For example:

```
const REG_ADDR: *const u8 = 0x5f3759df as *const u8;

let reg_value = unsafe { *REG_ADDR };
```
"##,

E0397: r##"
It is not allowed for a mutable static to allocate or have destructors. For
example:

```compile_fail
// error: mutable statics are not allowed to have boxes
static mut FOO: Option<Box<usize>> = None;

// error: mutable statics are not allowed to have destructors
static mut BAR: Option<Vec<i32>> = None;
```
"##,

E0400: r##"
A user-defined dereference was attempted in an invalid context. Erroneous
code example:

```compile_fail
use std::ops::Deref;

struct A;

impl Deref for A {
    type Target = str;

    fn deref(&self)-> &str { "foo" }
}

const S: &'static str = &A;
// error: user-defined dereference operators are not allowed in constants

fn main() {
    let foo = S;
}
```

You cannot directly use a dereference operation whilst initializing a constant
or a static. To fix this error, restructure your code to avoid this dereference,
perhaps moving it inline:

```
use std::ops::Deref;

struct A;

impl Deref for A {
    type Target = str;

    fn deref(&self)-> &str { "foo" }
}

fn main() {
    let foo : &str = &A;
}
```
"##,

E0492: r##"
A borrow of a constant containing interior mutability was attempted. Erroneous
code example:

```compile_fail
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

const A: AtomicUsize = ATOMIC_USIZE_INIT;
static B: &'static AtomicUsize = &A;
// error: cannot borrow a constant which contains interior mutability, create a
//        static instead
```

A `const` represents a constant value that should never change. If one takes
a `&` reference to the constant, then one is taking a pointer to some memory
location containing the value. Normally this is perfectly fine: most values
can't be changed via a shared `&` pointer, but interior mutability would allow
it. That is, a constant value could be mutated. On the other hand, a `static` is
explicitly a single memory location, which can be mutated at will.

So, in order to solve this error, either use statics which are `Sync`:

```
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

static A: AtomicUsize = ATOMIC_USIZE_INIT;
static B: &'static AtomicUsize = &A; // ok!
```

You can also have this error while using a cell type:

```compile_fail
#![feature(const_fn)]

use std::cell::Cell;

const A: Cell<usize> = Cell::new(1);
const B: &'static Cell<usize> = &A;
// error: cannot borrow a constant which contains interior mutability, create
//        a static instead

// or:
struct C { a: Cell<usize> }

const D: C = C { a: Cell::new(1) };
const E: &'static Cell<usize> = &D.a; // error

// or:
const F: &'static C = &D; // error
```

This is because cell types do operations that are not thread-safe. Due to this,
they don't implement Sync and thus can't be placed in statics. In this
case, `StaticMutex` would work just fine, but it isn't stable yet:
https://doc.rust-lang.org/nightly/std/sync/struct.StaticMutex.html

However, if you still wish to use these types, you can achieve this by an unsafe
wrapper:

```
#![feature(const_fn)]

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
A type with a destructor was assigned to an invalid type of variable. Erroneous
code example:

```compile_fail
struct Foo {
    a: u32
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

const F : Foo = Foo { a : 0 };
// error: constants are not allowed to have destructors
static S : Foo = Foo { a : 0 };
// error: statics are not allowed to have destructors
```

To solve this issue, please use a type which does allow the usage of type with
destructors.
"##,

E0494: r##"
A reference of an interior static was assigned to another const/static.
Erroneous code example:

```compile_fail
struct Foo {
    a: u32
}

static S : Foo = Foo { a : 0 };
static A : &'static u32 = &S.a;
// error: cannot refer to the interior of another static, use a
//        constant instead
```

The "base" variable has to be a const if you want another static/const variable
to refer to one of its fields. Example:

```
struct Foo {
    a: u32
}

const S : Foo = Foo { a : 0 };
static A : &'static u32 = &S.a; // ok!
```
"##,

}

register_diagnostics! {
    E0472, // asm! is unsupported on this target
}
