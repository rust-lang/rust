// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)
// min-lldb-version: 310

// compile-flags:--debuginfo=1

// gdb-command:run
// lldb-command:run

// Nothing to do here really, just make sure it compiles. See issue #8513.
fn main() {
    let _ = |&:|();
    let _ = range(1u,3).map(|_| 5i);
}

