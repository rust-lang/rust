rustc is slowly moving towards the [Rust standard coding style][fmt];
at the moment, however, it follows a rather more *chaotic*
style. There are a few things that are always true.

[fmt]: https://github.com/rust-lang-nursery/fmt-rfcs

# The tidy script

Running `./x.py test` begins with a "tidy" step. This tidy step checks
that your source files meet some basic conventions.

<a name=copyright>

## Copyright notice

All files must begin with the following copyright notice:

```
// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
```

The year at the top is not meaningful: copyright protections are in
fact automatic from the moment of authorship. We do not typically edit
the years on existing files. When creating a new file, you can use the
current year if you like, but you don't have to.

## Line length

Lines should be at most 100 characters. It's even better if you can
keep things to 80.

**Ignoring the line length limit.** Sometimes -- in particular for
tests -- it can be necessary to exempt yourself from this limit. In
that case, you can add a comment towards the top of the file (after
the copyright notice) like so:

```
// ignore-tidy-linelength
```

## Tabs vs spaces

Prefer 4-space indent.

# Warnings and lints

In general, Rust crates 

# Policy on using crates from crates.io

It is allowed to use crates from crates.io, though external
dependencies should not be added gratuitously. All such crates must
have a suitably permissive license. There is an automatic check which
inspects the Cargo metadata to ensure this.

