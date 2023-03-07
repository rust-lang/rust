#[repr(u16)]
#[allow(dead_code)]
enum DeviceKind {
    Nil = 0,
}

#[repr(packed)]
#[allow(dead_code)]
struct DeviceInfo {
    endianness: u8,
    device_kind: DeviceKind,
}

fn main() {
    let _x = None::<(DeviceInfo, u8)>;
    let _y = None::<(DeviceInfo, u16)>;
    let _z = None::<(DeviceInfo, u64)>;
}
