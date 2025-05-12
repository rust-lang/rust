use std::cell::Cell;

fn main() {
    unsafe {
        let root0 = Cell::new(42);
        let wildcard = &root0 as *const Cell<i32> as usize as *const Cell<i32>;
        // empty the stack to unknown (via SRW reborrow from wildcard)
        let _ref0 = &*wildcard;
        // Do a non-SRW reborrow from wildcard to start building up a stack again.
        // Now new refs start being inserted at idx 0, pushing the unique_range up.
        let _refn = &*&*&*&*&*(wildcard.cast::<i32>());
        // empty the stack again, but this time with unique_range.start sitting at some high index.
        let _ref0 = &*wildcard;
        // and do a read which tries to clear the uniques
        wildcard.cast::<i32>().read();
    }
}
