#![allow(unused)]

#[repr(u16)]
enum DeviceKind {
    Nil = 0,
}

#[repr(C, packed)]
struct DeviceInfo {
    endianness: u8,
    device_kind: DeviceKind,
}

fn main() {
    // The layout of `Option<(DeviceInfo, u64)>` is funny: it uses the
    // `DeviceKind` enum as niche, so that is offset 1, but the niche type is u16!
    // So despite the type having alignment 8 and the field type alignment 2,
    // the actual alignment is 1.
    let x = None::<(DeviceInfo, u8)>;
    let y = None::<(DeviceInfo, u16)>;
    let z = None::<(DeviceInfo, u64)>;
    format!("{} {} {}", x.is_some(), y.is_some(), y.is_some());
}
