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

```compile_fail,E0010
#![feature(box_syntax)]

const CON : Box<i32> = box 0;
```
"##,

E0013: r##"
Static and const variables can refer to other const variables. But a const
variable cannot refer to a static variable. For example, `Y` cannot refer to
`X` here:

```compile_fail,E0013
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

```compile_fail,E0016
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

```compile_fail,E0017
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

```compile_fail,E0018
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
static MY_STATIC: u32 = 42;
static MY_STATIC_ADDR: &'static u32 = &MY_STATIC;
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

    const A: i32 = FOO.test(); // You can't call Test::func() here!
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

E0381: r##"
It is not allowed to use or capture an uninitialized variable. For example:

```compile_fail,E0381
fn main() {
    let x: i32;
    let y = x; // error, use of possibly uninitialized variable
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

E0384: r##"
This error occurs when an attempt is made to reassign an immutable variable.
For example:

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


E0394: r##"
A static was referred to by value by another static.

Erroneous code examples:

```compile_fail,E0394
static A: u32 = 0;
static B: u32 = A; // error: cannot refer to other statics by value, use the
                   //        address-of operator or a constant instead
```

A static cannot be referred by value. To fix this issue, either use a
constant:

```
const A: u32 = 0; // `A` is now a constant
static B: u32 = A; // ok!
```

Or refer to `A` by reference:

```
static A: u32 = 0;
static B: &'static u32 = &A; // ok!
```
"##,

E0395: r##"
The value assigned to a constant scalar must be known at compile time,
which is not the case when comparing raw pointers.

Erroneous code example:

```compile_fail,E0395
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

E0161: r##"
A value was moved. However, its size was not known at compile time, and only
values of a known size can be moved.

Erroneous code example:

```compile_fail
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

E0396: r##"
The value behind a raw pointer can't be determined at compile-time
(or even link-time), which means it can't be used in a constant
expression. Erroneous code example:

```compile_fail,E0396
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

E0492: r##"
A borrow of a constant containing interior mutability was attempted. Erroneous
code example:

```compile_fail,E0492
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

const A: AtomicUsize = ATOMIC_USIZE_INIT;
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
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

static A: AtomicUsize = ATOMIC_USIZE_INIT;
static B: &'static AtomicUsize = &A; // ok!
```

You can also have this error while using a cell type:

```compile_fail,E0492
#![feature(const_fn)]

use std::cell::Cell;

const A: Cell<usize> = Cell::new(1);
const B: &'static Cell<usize> = &A;
// error: cannot borrow a constant which may contain interior mutability,
//        create a static instead

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

```compile_fail,E0493
struct Foo {
    a: u32
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

const F : Foo = Foo { a : 0 };
// error: constants are not allowed to have destructors
static S : Foo = Foo { a : 0 };
// error: destructors in statics are an unstable feature
```

To solve this issue, please use a type which does allow the usage of type with
destructors.
"##,

E0494: r##"
A reference of an interior static was assigned to another const/static.
Erroneous code example:

```compile_fail,E0494
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

E0499: r##"
A variable was borrowed as mutable more than once. Erroneous code example:

```compile_fail,E0499
let mut i = 0;
let mut x = &mut i;
let mut a = &mut i;
// error: cannot borrow `i` as mutable more than once at a time
```

Please note that in rust, you can either have many immutable references, or one
mutable reference. Take a look at
https://doc.rust-lang.org/stable/book/references-and-borrowing.html for more
information. Example:


```
let mut i = 0;
let mut x = &mut i; // ok!

// or:
let mut i = 0;
let a = &i; // ok!
let b = &i; // still ok!
let c = &i; // still ok!
```
"##,

E0500: r##"
A borrowed variable was used in another closure. Example of erroneous code:

```compile_fail
fn you_know_nothing(jon_snow: &mut i32) {
    let nights_watch = || {
        *jon_snow = 2;
    };
    let starks = || {
        *jon_snow = 3; // error: closure requires unique access to `jon_snow`
                       //        but it is already borrowed
    };
}
```

In here, `jon_snow` is already borrowed by the `nights_watch` closure, so it
cannot be borrowed by the `starks` closure at the same time. To fix this issue,
you can put the closure in its own scope:

```
fn you_know_nothing(jon_snow: &mut i32) {
    {
        let nights_watch = || {
            *jon_snow = 2;
        };
    } // At this point, `jon_snow` is free.
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
    let nights_watch = || {
        jon_copy = 2;
    };
    let starks = || {
        *jon_snow = 3;
    };
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

Example of erroneous code:

```compile_fail,E0501
fn inside_closure(x: &mut i32) {
    // Actions which require unique access
}

fn outside_closure(x: &mut i32) {
    // Actions which require unique access
}

fn foo(a: &mut i32) {
    let bar = || {
        inside_closure(a)
    };
    outside_closure(a); // error: cannot borrow `*a` as mutable because previous
                        //        closure requires unique access.
}
```

To fix this error, you can place the closure in its own scope:

```
fn inside_closure(x: &mut i32) {}
fn outside_closure(x: &mut i32) {}

fn foo(a: &mut i32) {
    {
        let bar = || {
            inside_closure(a)
        };
    } // borrow on `a` ends.
    outside_closure(a); // ok!
}
```

Or you can pass the variable as a parameter to the closure:

```
fn inside_closure(x: &mut i32) {}
fn outside_closure(x: &mut i32) {}

fn foo(a: &mut i32) {
    let bar = |s: &mut i32| {
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
    let bar = || {
        inside_closure(a)
    };
}
```
"##,

E0502: r##"
This error indicates that you are trying to borrow a variable as mutable when it
has already been borrowed as immutable.

Example of erroneous code:

```compile_fail,E0502
fn bar(x: &mut i32) {}
fn foo(a: &mut i32) {
    let ref y = a; // a is borrowed as immutable.
    bar(a); // error: cannot borrow `*a` as mutable because `a` is also borrowed
            //        as immutable
}
```

To fix this error, ensure that you don't have any other references to the
variable before trying to access it mutably:

```
fn bar(x: &mut i32) {}
fn foo(a: &mut i32) {
    bar(a);
    let ref y = a; // ok!
}
```

For more information on the rust ownership system, take a look at
https://doc.rust-lang.org/stable/book/references-and-borrowing.html.
"##,

E0503: r##"
A value was used after it was mutably borrowed.

Example of erroneous code:

```compile_fail,E0503
fn main() {
    let mut value = 3;
    // Create a mutable borrow of `value`. This borrow
    // lives until the end of this function.
    let _borrow = &mut value;
    let _sum = value + 1; // error: cannot use `value` because
                          //        it was mutably borrowed
}
```

In this example, `value` is mutably borrowed by `borrow` and cannot be
used to calculate `sum`. This is not possible because this would violate
Rust's mutability rules.

You can fix this error by limiting the scope of the borrow:

```
fn main() {
    let mut value = 3;
    // By creating a new block, you can limit the scope
    // of the reference.
    {
        let _borrow = &mut value; // Use `_borrow` inside this block.
    }
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
    let _borrow = &mut value;
    // ... which means we can still use `value_cloned`,
    let _sum = value_cloned + 1;
    // even though the borrow only ends here.
}
```

You can find more information about borrowing in the rust-book:
http://doc.rust-lang.org/stable/book/references-and-borrowing.html
"##,

E0504: r##"
This error occurs when an attempt is made to move a borrowed variable into a
closure.

Example of erroneous code:

```compile_fail,E0504
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

fn eat(val: Value) {}

fn main() {
    let x = Value{};
    {
        let _ref_to_val: &Value = &x;
        eat(x);
    }
}
```

Here, the function `eat` takes the ownership of `x`. However,
`x` cannot be moved because it was borrowed to `_ref_to_val`.
To fix that you can do few different things:

* Try to avoid moving the variable.
* Release borrow before move.
* Implement the `Copy` trait on the type.

Examples:

```
struct Value {}

fn eat(val: &Value) {}

fn main() {
    let x = Value{};
    {
        let _ref_to_val: &Value = &x;
        eat(&x); // pass by reference, if it's possible
    }
}
```

Or:

```
struct Value {}

fn eat(val: Value) {}

fn main() {
    let x = Value{};
    {
        let _ref_to_val: &Value = &x;
    }
    eat(x); // release borrow and then move it.
}
```

Or:

```
#[derive(Clone, Copy)] // implement Copy trait
struct Value {}

fn eat(val: Value) {}

fn main() {
    let x = Value{};
    {
        let _ref_to_val: &Value = &x;
        eat(x); // it will be copied here.
    }
}
```

You can find more information about borrowing in the rust-book:
http://doc.rust-lang.org/stable/book/references-and-borrowing.html
"##,

E0506: r##"
This error occurs when an attempt is made to assign to a borrowed value.

Example of erroneous code:

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

}

register_diagnostics! {
    E0524, // two closures require unique access to `..` at the same time
    E0526, // shuffle indices are not constant
    E0625, // thread-local statics cannot be accessed at compile-time
}
