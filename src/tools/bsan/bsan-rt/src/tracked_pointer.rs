use crate::*;

#[repr(C)]
#[derive(Debug)]
pub struct TrackedPointer {
    addr: *mut c_void,
    alloc_id: u64,
    tag: u64,
}

impl TrackedPointer {
    fn to_miri_pointer(pointer: TrackedPointer) -> Pointer {
        let addr = Size::from_bytes(pointer.addr as u64);
        let alloc_id = pointer.alloc_id;
        let tag = pointer.tag;
        if addr == Size::ZERO {
            Pointer::null()
        } else {
            let provenance = if alloc_id == 0 {
                None
            } else {
                let extra = if tag == 0 {
                    Provenance::Wildcard
                } else {
                    let alloc_id = unsafe { AllocId(NonZero::new_unchecked(alloc_id)) };
                    let tag = BorTag::new(tag).unwrap();
                    unsafe { Provenance::Concrete { alloc_id, tag } }
                };
                Some(extra)
            };
            Pointer::new(provenance, addr)
        }
    }
}
