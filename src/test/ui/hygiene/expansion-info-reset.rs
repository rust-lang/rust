// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: Investigate why expansion info for a single expansion id is reset from
// `MacroBang(format_args)` to `MacroAttribute(derive(Clone))` (issue #52363).

fn main() {
    format_args!({ #[derive(Clone)] struct S; });
    //~^ ERROR format argument must be a string literal
}
