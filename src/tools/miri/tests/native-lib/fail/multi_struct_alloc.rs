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
    let vals = [10u8, 20u8];
    let structs =
        vec![HasPointer { ptr: &raw const vals[0] }, HasPointer { ptr: &raw const vals[1] }];
    unsafe {
        access_struct_ptr(structs[1]);
        let _val = *std::ptr::with_exposed_provenance::<u8>(structs[0].ptr.addr()); //~ ERROR: Undefined Behavior: attempting a read access using <wildcard>
    };
}
