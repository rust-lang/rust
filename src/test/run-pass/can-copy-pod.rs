/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

// Tests that type parameters with the `Pod` are implicitly copyable.

#[allow(dead_code)];

fn can_copy_pod<T:Pod>(v: T) {
    let _a = v;
    let _b = v;
}

pub fn main() {}


