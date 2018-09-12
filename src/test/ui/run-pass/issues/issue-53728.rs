// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#[repr(u16)]
enum DeviceKind {
    Nil = 0,
}

#[repr(packed)]
struct DeviceInfo {
    endianness: u8,
    device_kind: DeviceKind,
}

fn main() {
    let _x = None::<(DeviceInfo, u8)>;
    let _y = None::<(DeviceInfo, u16)>;
    let _z = None::<(DeviceInfo, u64)>;
}
