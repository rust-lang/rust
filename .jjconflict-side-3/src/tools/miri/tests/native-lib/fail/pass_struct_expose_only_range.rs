//@compile-flags: -Zmiri-permissive-provenance

#[repr(C)]
#[derive(Copy, Clone)]
struct HasPointer {
    ptr: *const u8,
}

extern "C" {
    fn access_struct_ptr(s: HasPointer) -> u8;
}

fn main() {
    let structs = vec![HasPointer { ptr: &0 }, HasPointer { ptr: &1 }];
    unsafe {
        let r = access_struct_ptr(structs[1]);
        assert_eq!(r, 1);
        // There are two pointers stored in the allocation backing `structs`; ensure
        // we only exposed the one that was actually passed to C.
        let _val = *std::ptr::with_exposed_provenance::<u8>(structs[1].ptr.addr()); // fine, ptr got sent to C
        let _val = *std::ptr::with_exposed_provenance::<u8>(structs[0].ptr.addr()); //~ ERROR: memory access failed
    };
}
