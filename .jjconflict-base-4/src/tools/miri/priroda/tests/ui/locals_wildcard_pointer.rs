//@ normalize-stderr-test: "(?s)^warning: integer-to-pointer cast.*warning: 1 warning emitted\n\n$" -> ""

#[repr(C)]
struct WildcardPointer {
    ptr: *const u8,
}

fn main() {
    // The pointer is never dereferenced: this keeps the wildcard provenance
    // available for the debugger renderer without requiring it to resolve.
    let ptr = 0x1234usize as *const u8;
    let wildcard = WildcardPointer { ptr };

    // Keep the local alive for inspection.
    std::hint::black_box(&wildcard);
}
