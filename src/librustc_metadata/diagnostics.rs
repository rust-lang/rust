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
E0454: r##"
A link name was given with an empty name. Erroneous code example:

```
#[link(name = "")] extern {} // error: #[link(name = "")] given with empty name
```

The rust compiler cannot link to an external library if you don't give it its
name. Example:

```
#[link(name = "some_lib")] extern {} // ok!
```
"##,

E0455: r##"
Linking with `kind=framework` is only supported when targeting OS X,
as frameworks are specific to that operating system.

Erroneous code example:

```compile_fail"
#[link(name = "FooCoreServices",  kind = "framework")] extern {}
// OS used to compile is Linux for example
```

To solve this error you can use conditional compilation:

```
#[cfg_attr(target="macos", link(name = "FooCoreServices", kind = "framework"))]
extern {}
```

See more: https://doc.rust-lang.org/book/conditional-compilation.html
"##,

E0458: r##"
An unknown "kind" was specified for a link attribute. Erroneous code example:

```
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

```
#[link(kind = "dylib")] extern {}
// error: #[link(...)] specified without `name = "foo"`
```

Please add the name parameter to allow the rust compiler to find the library
you want. Example:

```
#[link(kind = "dylib", name = "some_lib")] extern {} // ok!
```
"##,

E0463: r##"
A plugin/crate was declared but cannot be found. Erroneous code example:

```
#![feature(plugin)]
#![plugin(cookie_monster)] // error: can't find crate for `cookie_monster`
extern crate cake_is_a_lie; // error: can't find crate for `cake_is_a_lie`
```

You need to link your code to the relevant crate in order to be able to use it
(through Cargo or the `-L` option of rustc example). Plugins are crates as
well, and you link to them the same way.
"##,

E0467: r##"
Invalid or no macros listed for reexport.

Causes of this error:
```ignore
#[macro_reexport] // error: no macros listed for export
extern crate macros_for_good;
```

```ignore
#[macro_reexport(fun_macro = "foo")] // error: not a macro identifier
extern crate macros_for_good;
```

Currently, `macro_reexport` requires at least one macro name to be listed. 
Unlike `macro_use`, listing no names does not reexport all macros from the 
given crate.

Decide which macros you would like to export and list them properly.

"##,

E0468: r##"
A non-root module attempts to import macros from another crate.

```ignore
mod foo {
    #[macro_use(helpful_macro)] // error: must be at crate root to import
    extern crate some_crate;    //        macros from another crate
    helpful_macro!(...)
}

fn main() {
    // ...
}
```

Only `extern crate` imports at the crate root level (i.e., in lib.rs) are 
allowed to import macros.

Either move the macro import to crate root or do without the foreign macros.

This will work: 

```ignore
#[macro_use(helpful_macro)]
extern crate some_crate;
mod foo {
    helpful_macro!(...)
}

fn main() {
    //...
}
```

"##,

E0469: r##"
A macro listed for import was not found.

```ignore
/// // crate some_crate contains:
/// macro_rules! eat {
///     ...
/// }
/// macro_rules! drink {
///     ...
/// }
#[macro_use(be_merry)]      // error: be_merry is not
extern crate some_crate;    // a macro in some_crate!
```

This is likely caused by a typo. Did you misspell the macro's name?

Double-check the names of the macros listed for import, and that the crate 
in question exports them.
"##,

E0470: r##"
A macro listed for reexport was not found.

```ignore
/// // crate some_crate contains:
/// macro_rules! eat {
///     ...
/// }
/// macro_rules! drink {
///     ...
/// }
#[macro_reexport(be_merry)] // error: be_merry is not
extern crate some_crate;    // a macro in some_crate!
```

This is likely caused by a typo. Did you misspell the macro's name?

Double-check the names of the macros listed for reexport, and that the crate 
in question exports them.
"##
}

register_diagnostics! {
    E0456, // plugin `..` is not available for triple `..`
    E0457, // plugin `..` only found in rlib format, but must be available...
    E0514, // metadata version mismatch
    E0460, // found possibly newer version of crate `..`
    E0461, // couldn't find crate `..` with expected target triple ..
    E0462, // found staticlib `..` instead of rlib or dylib
    E0464, // multiple matching crates for `..`
    E0465, // multiple .. candidates for `..` found
    E0466, // bad macro import
    E0519, // local crate and dependency have same (crate-name, disambiguator)
    E0523, // two dependencies have same (crate-name, disambiguator) but different SVH
}
