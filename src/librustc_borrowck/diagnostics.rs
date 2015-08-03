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
"##

}

register_diagnostics! {
    E0382, // use of partially/collaterally moved value
    E0383, // partial reinitialization of uninitialized structure
    E0385, // {} in an aliasable location
    E0386, // {} in an immutable container
    E0387, // {} in a captured outer variable in an `Fn` closure
    E0388, // {} in a static location
    E0389  // {} in a `&` reference
}
