pub(super) trait SpecFill<T> {
    fn spec_fill(&mut self, value: T);
}

impl<T: Clone> SpecFill<T> for [T] {
    default fn spec_fill(&mut self, value: T) {
        if let Some((last, elems)) = self.split_last_mut() {
            for el in elems {
                el.clone_from(&value);
            }

            *last = value
        }
    }
}

impl<T: Copy> SpecFill<T> for [T] {
    default fn spec_fill(&mut self, value: T) {
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

impl SpecFill<u8> for [u8] {
    fn spec_fill(&mut self, value: u8) {
        // SAFETY: The pointer is derived from a reference, so it's writable.
        unsafe {
            crate::intrinsics::write_bytes(self.as_mut_ptr(), value, self.len());
        }
    }
}

impl SpecFill<i8> for [i8] {
    fn spec_fill(&mut self, value: i8) {
        // SAFETY: The pointer is derived from a reference, so it's writable.
        unsafe {
            crate::intrinsics::write_bytes(self.as_mut_ptr(), value.cast_unsigned(), self.len());
        }
    }
}

macro spec_fill_int {
    ($($type:ty)*) => {$(
        impl SpecFill<$type> for [$type] {
            #[inline]
            fn spec_fill(&mut self, value: $type) {
                // We always take this fastpath in Miri for long slices as the manual `for`
                // loop can be prohibitively slow.
                if (cfg!(miri) && self.len() > 32) || crate::intrinsics::is_val_statically_known(value) {
                    let bytes = value.to_ne_bytes();
                    if value == <$type>::from_ne_bytes([bytes[0]; size_of::<$type>()]) {
                        // SAFETY: The pointer is derived from a reference, so it's writable.
                        unsafe {
                            crate::intrinsics::write_bytes(self.as_mut_ptr(), bytes[0], self.len());
                        }
                        return;
                    }
                }
                for item in self.iter_mut() {
                    *item = value;
                }
            }
        }
    )*}
}

spec_fill_int! { u16 i16 u32 i32 u64 i64 u128 i128 usize isize }
