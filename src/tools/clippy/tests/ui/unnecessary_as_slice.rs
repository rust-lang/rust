#![warn(clippy::unnecessary_as_slice)]
#![allow(clippy::const_is_empty, clippy::ptr_arg)]
#![expect(unused)]

trait SliceExt {
    fn my_len(&self) -> usize;
}

impl SliceExt for [u32] {
    fn my_len(&self) -> usize {
        self.len()
    }
}

fn by_ref(vals: &mut Vec<u32>) {
    let _ = vals.as_slice().len();
    //~^ unnecessary_as_slice
    let _ = vals.as_slice().is_empty();
    //~^ unnecessary_as_slice
    let _ = vals.as_slice().iter();
    //~^ unnecessary_as_slice
    let _ = vals.as_slice().first();
    //~^ unnecessary_as_slice

    let _ = vals.as_mut_slice().len();
    //~^ unnecessary_as_slice
    let _ = vals.as_mut_slice().is_empty();
    //~^ unnecessary_as_slice
    let _ = vals.as_mut_slice().iter_mut();
    //~^ unnecessary_as_slice
}

fn by_value(mut vals: Vec<u32>) {
    let _ = vals.as_slice().len();
    //~^ unnecessary_as_slice
    let _ = vals.as_mut_slice().len();
    //~^ unnecessary_as_slice
}

fn custom_trait_method(mut vals: Vec<u32>) {
    let _ = vals.as_slice().my_len();
    let _ = vals.as_mut_slice().my_len();
}

fn no_lint_cases(vals: &mut Vec<u32>) {
    // Don't lint as_slice() received by a value
    let s = vals.as_slice();
    let _ = s.len();

    fn takes_slice(_: &[u32]) {}
    takes_slice(vals.as_slice());

    let _ = vals.len();
    let _ = vals.is_empty();

    fn non_recv(_: &[u32]) {}
    // Don't lint as_slice() when it's not the receiver of the method call
    non_recv(vals.as_slice());

    // Don't lint if as_slice() is in a macro.
    macro_rules! as_slice_in_macro {
        ($vals:expr) => {
            let _ = $vals.as_slice().len();
            let _ = $vals.as_mut_slice().len();
        };
    }

    as_slice_in_macro!(vals);

    trait NoLintTrait {
        fn foo(&self) -> usize;
        fn bar(&mut self) -> usize;
    }

    impl NoLintTrait for Vec<u32> {
        fn foo(&self) -> usize {
            1
        }
        fn bar(&mut self) -> usize {
            2
        }
    }

    impl NoLintTrait for [u32] {
        fn foo(&self) -> usize {
            3
        }
        fn bar(&mut self) -> usize {
            4
        }
    }

    // Don't lint trait method calls.
    let _ = vals.as_slice().foo();
    let _ = vals.as_mut_slice().bar();
}

fn no_lint_non_vec(vals: &[u32]) {
    // vec_as_slice diagnostic item should only applies to Vec::as_slice
    let _ = vals.len();
    let _ = vals.is_empty();
}

// Don't lint if the parent method is `as_ptr` or `as_mut_ptr`
fn no_lint_as_ptr(vals: &mut Vec<u32>) {
    let _ = vals.as_ptr();
    let _ = vals.as_mut_ptr();

    let _ = vals.as_slice().as_ptr();
    let _ = vals.as_mut_slice().as_mut_ptr();
}

fn main() {}
