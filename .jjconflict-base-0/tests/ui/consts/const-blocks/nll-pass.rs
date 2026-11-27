//@ check-pass
#![allow(warnings)]

// Some type that is not copyable.
struct Bar;

mod constants {
    use crate::Bar;

    fn no_impl_copy_empty_value_no_elements() {
        const FOO: Option<Bar> = None;
        const ARR: [Option<Bar>; 0] = [FOO; 0];
    }

    fn no_impl_copy_empty_value_single_element() {
        const FOO: Option<Bar> = None;
        const ARR: [Option<Bar>; 1] = [FOO; 1];
    }

    fn no_impl_copy_empty_value_multiple_elements() {
        const FOO: Option<Bar> = None;
        const ARR: [Option<Bar>; 2] = [FOO; 2];
    }

    fn no_impl_copy_value_no_elements() {
        const FOO: Option<Bar> = Some(Bar);
        const ARR: [Option<Bar>; 0] = [FOO; 0];
    }

    fn no_impl_copy_value_single_element() {
        const FOO: Option<Bar> = Some(Bar);
        const ARR: [Option<Bar>; 1] = [FOO; 1];
    }

    fn no_impl_copy_value_multiple_elements() {
        const FOO: Option<Bar> = Some(Bar);
        const ARR: [Option<Bar>; 2] = [FOO; 2];
    }

    fn impl_copy_empty_value_no_elements() {
        const FOO: Option<u32> = None;
        const ARR: [Option<u32>; 0] = [FOO; 0];
    }

    fn impl_copy_empty_value_one_element() {
        const FOO: Option<u32> = None;
        const ARR: [Option<u32>; 1] = [FOO; 1];
    }

    fn impl_copy_empty_value_multiple_elements() {
        const FOO: Option<u32> = None;
        const ARR: [Option<u32>; 2] = [FOO; 2];
    }

    fn impl_copy_value_no_elements() {
        const FOO: Option<u32> = Some(4);
        const ARR: [Option<u32>; 0] = [FOO; 0];
    }

    fn impl_copy_value_one_element() {
        const FOO: Option<u32> = Some(4);
        const ARR: [Option<u32>; 1] = [FOO; 1];
    }

    fn impl_copy_value_multiple_elements() {
        const FOO: Option<u32> = Some(4);
        const ARR: [Option<u32>; 2] = [FOO; 2];
    }
}

mod non_constants {
    use crate::Bar;

    fn no_impl_copy_empty_value_no_elements() {
        let x = None;
        let arr: [Option<Bar>; 0] = [x; 0];
    }

    fn no_impl_copy_empty_value_single_element() {
        let x = None;
        let arr: [Option<Bar>; 1] = [x; 1];
    }

    fn no_impl_copy_value_no_elements() {
        let x = Some(Bar);
        let arr: [Option<Bar>; 0] = [x; 0];
    }

    fn no_impl_copy_value_single_element() {
        let x = Some(Bar);
        let arr: [Option<Bar>; 1] = [x; 1];
    }

    fn impl_copy_empty_value_no_elements() {
        let x: Option<u32> = None;
        let arr: [Option<u32>; 0] = [x; 0];
    }

    fn impl_copy_empty_value_one_element() {
        let x: Option<u32> = None;
        let arr: [Option<u32>; 1] = [x; 1];
    }

    fn impl_copy_empty_value_multiple_elements() {
        let x: Option<u32> = None;
        let arr: [Option<u32>; 2] = [x; 2];
    }

    fn impl_copy_value_no_elements() {
        let x: Option<u32> = Some(4);
        let arr: [Option<u32>; 0] = [x; 0];
    }

    fn impl_copy_value_one_element() {
        let x: Option<u32> = Some(4);
        let arr: [Option<u32>; 1] = [x; 1];
    }

    fn impl_copy_value_multiple_elements() {
        let x: Option<u32> = Some(4);
        let arr: [Option<u32>; 2] = [x; 2];
    }
}

fn main() {}
