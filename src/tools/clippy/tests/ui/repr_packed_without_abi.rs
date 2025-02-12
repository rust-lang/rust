#![deny(clippy::repr_packed_without_abi)]

#[repr(packed)]
struct NetworkPacketHeader {
    header_length: u8,
    header_version: u16,
}

#[repr(packed)]
union Foo {
    a: u8,
    b: u16,
}

#[repr(C, packed)]
struct NoLintCNetworkPacketHeader {
    header_length: u8,
    header_version: u16,
}

#[repr(Rust, packed)]
struct NoLintRustNetworkPacketHeader {
    header_length: u8,
    header_version: u16,
}

#[repr(packed, C)]
union NotLintCFoo {
    a: u8,
    b: u16,
}

#[repr(packed, Rust)]
union NotLintRustFoo {
    a: u8,
    b: u16,
}
