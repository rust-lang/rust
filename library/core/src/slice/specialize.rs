use crate::mem::{size_of, transmute_copy};
use crate::ptr::write_bytes;

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
    fn spec_fill(&mut self, value: T) {
        if size_of::<T>() == 1 {
            // SAFETY: The size_of check above ensures that values are 1 byte wide, as required
            // for the transmute and write_bytes
            unsafe {
                let value: u8 = transmute_copy(&value);
                write_bytes(self.as_mut_ptr(), value, self.len());
            }
        } else {
            for item in self.iter_mut() {
                *item = value;
            }
        }
    }
}
