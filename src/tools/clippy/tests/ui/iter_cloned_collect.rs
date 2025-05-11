#![allow(unused)]
#![allow(clippy::useless_vec)]

use std::collections::{HashSet, VecDeque};

fn main() {
    let v = [1, 2, 3, 4, 5];
    let v2: Vec<isize> = v.iter().cloned().collect();
    //~^ iter_cloned_collect
    let v3: HashSet<isize> = v.iter().cloned().collect();
    let v4: VecDeque<isize> = v.iter().cloned().collect();

    // Handle macro expansion in suggestion
    let _: Vec<isize> = vec![1, 2, 3].iter().cloned().collect();
    //~^ iter_cloned_collect

    // Issue #3704
    unsafe {
        let _: Vec<u8> = std::ffi::CStr::from_ptr(std::ptr::null())
            .to_bytes()
            //~^ iter_cloned_collect
            .iter()
            .cloned()
            .collect();
    }

    // Issue #6808
    let arr: [u8; 64] = [0; 64];
    let _: Vec<_> = arr.iter().cloned().collect();
    //~^ iter_cloned_collect

    // Issue #6703
    let _: Vec<isize> = v.iter().copied().collect();
    //~^ iter_cloned_collect
}

mod issue9119 {

    use std::iter;

    #[derive(Clone)]
    struct Example(u16);

    impl iter::FromIterator<Example> for Vec<u8> {
        fn from_iter<T>(iter: T) -> Self
        where
            T: IntoIterator<Item = Example>,
        {
            iter.into_iter().flat_map(|e| e.0.to_le_bytes()).collect()
        }
    }

    fn foo() {
        let examples = [Example(1), Example(0x1234)];
        let encoded: Vec<u8> = examples.iter().cloned().collect();
        assert_eq!(encoded, vec![0x01, 0x00, 0x34, 0x12]);

        let a = [&&String::new()];
        let v: Vec<&&String> = a.iter().cloned().collect();
        //~^ iter_cloned_collect
    }
}
