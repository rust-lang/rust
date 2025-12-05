//@revisions: stack tree
//@compile-flags: -Zmiri-strict-provenance
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::collections::VecDeque;

fn test_all_refs<'a, T: 'a>(dummy: &mut T, iter: impl Iterator<Item = &'a mut T>) {
    // Gather all those references.
    let mut refs: Vec<&mut T> = iter.collect();
    // Use them all. Twice, to be sure we got all interleavings.
    for r in refs.iter_mut() {
        std::mem::swap(dummy, r);
    }
    for r in refs {
        std::mem::swap(dummy, r);
    }
}

fn main() {
    let mut dst = VecDeque::new();
    dst.push_front(Box::new(1));
    dst.push_front(Box::new(2));
    dst.pop_back();

    let mut src = VecDeque::new();
    src.push_front(Box::new(2));
    dst.append(&mut src);
    for a in dst.iter() {
        assert_eq!(**a, 2);
    }

    // Regression test for Debug impl's
    let _ = format!("{:?} {:?}", dst, dst.iter());
    let _ = format!("{:?}", VecDeque::<u32>::new().iter());

    for a in dst {
        assert_eq!(*a, 2);
    }

    // # Aliasing tests.
    let mut v = std::collections::VecDeque::new();
    v.push_back(1);
    v.push_back(2);

    // Test `fold` bad aliasing.
    let mut it = v.iter_mut();
    let ref0 = it.next().unwrap();
    let sum = it.fold(0, |x, y| x + *y);
    assert_eq!(*ref0 + sum, 3);

    // Test general iterator aliasing.
    v.push_front(0);
    test_all_refs(&mut 0, v.iter_mut());
}
