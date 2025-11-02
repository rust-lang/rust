// https://github.com/rust-lang/rust/issues/53728
//@ run-pass

#![allow(dead_code)]
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
