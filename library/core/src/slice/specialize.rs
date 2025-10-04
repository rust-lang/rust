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
        if size_of::<T>() == 1 {
            // SAFETY: The pointer is derived from a reference, so it's writable.
            // And we checked that T is 1 byte wide.
            unsafe {
                // use the intrinsic since it allows any T as long as it's 1 byte wide
                crate::intrinsics::write_bytes(self.as_mut_ptr(), value, self.len());
            }
            return;
        }
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

macro spec_fill_int {
    ($($type:ty)*) => {$(
        impl SpecFill<$type> for [$type] {
            #[inline]
            fn spec_fill(&mut self, value: $type) {
                if crate::intrinsics::is_val_statically_known(value) {
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
