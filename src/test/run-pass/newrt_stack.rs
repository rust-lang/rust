// xfail-test not a test

pub struct StackSegment {
    buf: ~[u8]
}

impl StackSegment {
    static fn new(size: uint) -> StackSegment {
        // Crate a block of uninitialized values
        let mut stack = vec::with_capacity(size);
        unsafe {
            vec::raw::set_len(&mut stack, size);
        }

        StackSegment {
            buf: stack
        }
    }

    fn end(&self) -> *uint {
        unsafe {
            vec::raw::to_ptr(self.buf).offset(self.buf.len()) as *uint
        }
    }
}

pub struct StackPool(());

impl StackPool {

    static fn new() -> StackPool { StackPool(()) }

    fn take_segment(&self, min_size: uint) -> StackSegment {
        StackSegment::new(min_size)
    }

    fn give_segment(&self, _stack: StackSegment) {
    }
}
