// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-C panic=abort -C prefer-dynamic
// ignore-musl - no dylibs here
// ignore-emscripten
// error-pattern:`panic_unwind` is not compiled with this crate's panic strategy

// This is a test where the local crate, compiled with `panic=abort`, links to
// the standard library **dynamically** which is already linked against
// `panic=unwind`. We should fail because the linked panic runtime does not
// correspond with our `-C panic` option.
//
// Note that this test assumes that the dynamic version of the standard library
// is linked to `panic_unwind`, which is currently the case.

fn main() {
}
