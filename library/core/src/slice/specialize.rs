use crate::clone::TrivialClone;
use crate::ptr;

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

impl<T: TrivialClone> SpecFill<T> for [T] {
    fn spec_fill(&mut self, value: T) {
        for item in self.iter_mut() {
            // SAFETY: `TrivialClone` indicates that this is equivalent to
            // calling `Clone::clone`
            *item = unsafe { ptr::read(&value) };
        }
    }
}
