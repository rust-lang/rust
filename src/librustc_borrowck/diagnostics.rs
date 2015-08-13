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

E0373: r##"
This error occurs when an attempt is made to use data captured by a closure,
when that data may no longer exist. It's most commonly seen when attempting to
return a closure:

```
fn foo() -> Box<Fn(u32) -> u32> {
    let x = 0u32;
    Box::new(|y| x + y)
}
```

Notice that `x` is stack-allocated by `foo()`. By default, Rust captures
closed-over data by reference. This means that once `foo()` returns, `x` no
longer exists. An attempt to access `x` within the closure would thus be unsafe.

Another situation where this might be encountered is when spawning threads:

```
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

E0381: r##"
It is not allowed to use or capture an uninitialized variable. For example:

```
fn main() {
    let x: i32;
    let y = x; // error, use of possibly uninitialized variable
```

To fix this, ensure that any declared variables are initialized before being
used.
"##,

E0382: r##"
This error occurs when an attempt is made to use a variable after its contents
have been moved elsewhere. For example:

```
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

If we own the type, the easiest way to address this problem is to implement
`Copy` and `Clone` on it, as shown below. This allows `y` to copy the
information in `x`, while leaving the original version owned by `x`. Subsequent
changes to `x` will not be reflected when accessing `y`.

```
#[derive(Copy, Clone)]
struct MyStruct { s: u32 }

fn main() {
    let mut x = MyStruct{ s: 5u32 };
    let y = x;
    x.s = 6;
    println!("{}", x.s);
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
    println!("{}", x.borrow.s);
}
```

With this approach, x and y share ownership of the data via the `Rc` (reference
count type). `RefCell` essentially performs runtime borrow checking: ensuring
that at most one writer or multiple readers can access the data at any one time.

If you wish to learn more about ownership in Rust, start with the chapter in the
Book:

https://doc.rust-lang.org/book/ownership.html
"##,

E0383: r##"
This error occurs when an attempt is made to partially reinitialize a
structure that is currently uninitialized.

For example, this can happen when a drop has taken place:

```
let mut x = Foo { a: 1 };
drop(x); // `x` is now uninitialized
x.a = 2; // error, partial reinitialization of uninitialized structure `t`
```

This error can be fixed by fully reinitializing the structure in question:

```
let mut x = Foo { a: 1 };
drop(x);
x = Foo { a: 2 };
```
"##,

E0384: r##"
This error occurs when an attempt is made to reassign an immutable variable.
For example:

```
fn main(){
    let x = 3;
    x = 5; // error, reassignment of immutable variable
}
```

By default, variables in Rust are immutable. To fix this error, add the keyword
`mut` after the keyword `let` when declaring the variable. For example:

```
fn main(){
    let mut x = 3;
    x = 5;
}
```
"##,

E0386: r##"
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

It can also be fixed by using a type with interior mutability, such as `Cell` or
`RefCell`:

```
let x: i64 = 1;
let y: Box<Cell<_>> = Box::new(Cell::new(x));
y.set(2);
```
"##,

E0387: r##"
This error occurs when an attempt is made to mutate or mutably reference data
that a closure has captured immutably. Examples of this error are shown below:

```
// Accepts a function or a closure that captures its environment immutably.
// Closures passed to foo will not be able to mutate their closed-over state.
fn foo<F: Fn()>(f: F) { }

// Attempts to mutate closed-over data.  Error message reads:
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
interior mutability through a shared reference. Our example's `mutable` function
could be redefined as below:

```
use std::cell::Cell;

fn mutable() {
    let x = Cell::new(0u32);
    foo(|| x.set(2));
}
```

You can read more about cell types in the API documentation:

https://doc.rust-lang.org/std/cell/
"##

}

register_diagnostics! {
    E0385, // {} in an aliasable location
    E0388, // {} in a static location
    E0389  // {} in a `&` reference
}
