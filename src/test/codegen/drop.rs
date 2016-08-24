// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

struct SomeUniqueName;

impl Drop for SomeUniqueName {
    fn drop(&mut self) {
    }
}

pub fn possibly_unwinding() {
}

// CHECK-LABEL: @droppy
#[no_mangle]
pub fn droppy() {
// Check that there are exactly 6 drop calls. The cleanups for the unwinding should be reused, so
// that's one new drop call per call to possibly_unwinding(), and finally 3 drop calls for the
// regular function exit. We used to have problems with quadratic growths of drop calls in such
// functions.
// CHECK-NOT: invoke{{.*}}drop{{.*}}SomeUniqueName
// CHECK: call{{.*}}drop{{.*}}SomeUniqueName
// CHECK: call{{.*}}drop{{.*}}SomeUniqueName
// CHECK: call{{.*}}drop{{.*}}SomeUniqueName
// CHECK-NOT: call{{.*}}drop{{.*}}SomeUniqueName
// CHECK: invoke{{.*}}drop{{.*}}SomeUniqueName
// CHECK: invoke{{.*}}drop{{.*}}SomeUniqueName
// CHECK: invoke{{.*}}drop{{.*}}SomeUniqueName
// CHECK-NOT: {{(call|invoke).*}}drop{{.*}}SomeUniqueName
// The next line checks for the } that ends the function definition
// CHECK-LABEL: {{^[}]}}
    let _s = SomeUniqueName;
    possibly_unwinding();
    let _s = SomeUniqueName;
    possibly_unwinding();
    let _s = SomeUniqueName;
    possibly_unwinding();
}
