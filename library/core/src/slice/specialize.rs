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
    default fn spec_fill(&mut self, value: T) {
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

impl SpecFill<u8> for [u8] {
    fn spec_fill(&mut self, value: u8) {
        // SAFETY: this is slice of u8
        unsafe {
            let ptr = self.as_mut_ptr();
            let len = self.len();
            write_bytes(ptr, value, len);
        }
    }
}

impl SpecFill<i8> for [i8] {
    fn spec_fill(&mut self, value: i8) {
        // SAFETY: this is slice of i8
        unsafe {
            let ptr = self.as_mut_ptr();
            let len = self.len();
            write_bytes(ptr, value as u8, len);
        }
    }
}

impl SpecFill<bool> for [bool] {
    fn spec_fill(&mut self, value: bool) {
        // SAFETY: this is slice of bool
        unsafe {
            let ptr = self.as_mut_ptr();
            let len = self.len();
            write_bytes(ptr, value as u8, len);
        }
    }
}
