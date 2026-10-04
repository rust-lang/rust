use std::{ptr::NonNull, task::Poll};

struct TaskRef;

struct Header {
    vtable: &'static Vtable,
}

struct Vtable {
    poll: unsafe fn(TaskRef) -> Poll<()>,
    deallocate: unsafe fn(NonNull<Header>),
}

// in the "Header" type, which is a private type in maitake
impl Header {
    pub(crate) const fn new_stub() -> Self {
        unsafe fn nop(_ptr: TaskRef) -> Poll<()> {
            Poll::Pending
        }

        unsafe fn nop_deallocate(ptr: NonNull<Header>) {
            unreachable!("stub task ({ptr:p}) should never be deallocated!");
        }

        Self { vtable: &Vtable { poll: nop, deallocate: nop_deallocate } }
    }
}

// This is a public type in `maitake`
#[repr(transparent)]
#[cfg_attr(loom, allow(dead_code))]
pub struct TaskStub {
    hdr: Header,
}

impl TaskStub {
    /// Create a new unique stub [`Task`].
    pub const fn new() -> Self {
        Self { hdr: Header::new_stub() }
    }
}
