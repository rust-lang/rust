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

}

register_diagnostics! {
    E0455, // native frameworks are only available on OSX targets
    E0456, // plugin `..` is not available for triple `..`
    E0457, // plugin `..` only found in rlib format, but must be available...
    E0514, // metadata version mismatch
    E0460, // found possibly newer version of crate `..`
    E0461, // couldn't find crate `..` with expected target triple ..
    E0462, // found staticlib `..` instead of rlib or dylib
    E0463, // can't find crate for `..`
    E0464, // multiple matching crates for `..`
    E0465, // multiple .. candidates for `..` found
    E0466, // bad macro import
    E0467, // bad macro reexport
    E0468, // an `extern crate` loading macros must be at the crate root
    E0469, // imported macro not found
    E0470, // reexported macro not found
}
