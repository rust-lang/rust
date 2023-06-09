//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-strict-provenance
#![feature(iter_advance_by, iter_next_chunk)]

// Gather all references from a mutable iterator and make sure Miri notices if
// using them is dangerous.
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

fn make_vec() -> Vec<u8> {
    let mut v = Vec::with_capacity(4);
    v.push(1);
    v.push(2);
    v
}

fn make_vec_macro() -> Vec<u8> {
    vec![1, 2]
}

fn make_vec_macro_repeat() -> Vec<u8> {
    vec![42; 5]
}

fn make_vec_macro_repeat_zeroed() -> Vec<u8> {
    vec![0; 7]
}

fn vec_into_iter() -> u8 {
    vec![1, 2, 3, 4].into_iter().map(|x| x * x).fold(0, |x, y| x + y)
}

fn vec_into_iter_rev() -> u8 {
    vec![1, 2, 3, 4].into_iter().rev().map(|x| x * x).fold(0, |x, y| x + y)
}

fn vec_into_iter_zst() {
    for _ in vec![[0u64; 0]].into_iter() {}
    let v = vec![[0u64; 0], [0u64; 0]].into_iter().map(|x| x.len()).sum::<usize>();
    assert_eq!(v, 0);

    let mut it = vec![[0u64; 0], [0u64; 0]].into_iter();
    it.advance_by(1).unwrap();
    drop(it);

    let mut it = vec![[0u64; 0], [0u64; 0]].into_iter();
    it.next_chunk::<1>().unwrap();
    drop(it);

    let mut it = vec![[0u64; 0], [0u64; 0]].into_iter();
    it.next_chunk::<4>().unwrap_err();
    drop(it);
}

fn vec_into_iter_rev_zst() {
    for _ in vec![[0u64; 0]; 5].into_iter().rev() {}
    let v = vec![[0u64; 0], [0u64; 0]].into_iter().rev().map(|x| x.len()).sum::<usize>();
    assert_eq!(v, 0);
}

fn vec_iter_and_mut() {
    let mut v = vec![1, 2, 3, 4];
    for i in v.iter_mut() {
        *i += 1;
    }
    assert_eq!(v.iter().sum::<i32>(), 2 + 3 + 4 + 5);

    test_all_refs(&mut 13, v.iter_mut());
}

fn vec_iter_and_mut_rev() {
    let mut v = vec![1, 2, 3, 4];
    for i in v.iter_mut().rev() {
        *i += 1;
    }
    assert_eq!(v.iter().sum::<i32>(), 2 + 3 + 4 + 5);
}

fn vec_reallocate() -> Vec<u8> {
    let mut v = vec![1, 2];
    v.push(3);
    v.push(4);
    v.push(5);
    v
}

fn vec_push_ptr_stable() {
    let mut v = Vec::with_capacity(10);
    v.push(0);
    let v0 = unsafe { &mut *(&mut v[0] as *mut _) }; // laundering the lifetime -- we take care that `v` does not reallocate, so that's okay.
    v.push(1);
    let _val = *v0;
}

fn vec_extend_ptr_stable() {
    let mut v = Vec::with_capacity(10);
    v.push(0);
    let v0 = unsafe { &mut *(&mut v[0] as *mut _) }; // laundering the lifetime -- we take care that `v` does not reallocate, so that's okay.
    // `slice::Iter` (with `T: Copy`) specialization
    v.extend(&[1]);
    let _val = *v0;
    // `vec::IntoIter` specialization
    v.extend(vec![2]);
    let _val = *v0;
    // `TrustedLen` specialization
    v.extend(std::iter::once(3));
    let _val = *v0;
    // base case
    v.extend(std::iter::once(3).filter(|_| true));
    let _val = *v0;
}

fn vec_truncate_ptr_stable() {
    let mut v = vec![0; 10];
    let v0 = unsafe { &mut *(&mut v[0] as *mut _) }; // laundering the lifetime -- we take care that `v` does not reallocate, so that's okay.
    v.truncate(5);
    let _val = *v0;
}

fn push_str_ptr_stable() {
    let mut buf = String::with_capacity(11);
    buf.push_str("hello");
    let hello: &str = unsafe { &*(buf.as_str() as *const _) }; // laundering the lifetime -- we take care that `buf` does not reallocate, so that's okay.
    buf.push_str(" world");
    assert_eq!(format!("{}", hello), "hello");
}

fn sort() {
    let mut v = vec![1; 20];
    v.push(0);
    v.sort();
}

fn swap() {
    let mut v = vec![1, 2, 3, 4];
    v.swap(2, 2);
}

fn swap_remove() {
    let mut a = 0;
    let mut b = 1;
    let mut vec = vec![&mut a, &mut b];

    vec.swap_remove(1);
}

fn reverse() {
    #[repr(align(2))]
    #[derive(Debug)]
    struct Foo(u8);

    let mut v: Vec<_> = (0..50).map(Foo).collect();
    v.reverse();
    assert!(v[0].0 == 49);
}

fn miri_issue_2759() {
    let mut input = "1".to_string();
    input.replace_range(0..0, "0");
}

fn main() {
    assert_eq!(vec_reallocate().len(), 5);

    assert_eq!(vec_into_iter(), 30);
    assert_eq!(vec_into_iter_rev(), 30);
    vec_iter_and_mut();
    vec_into_iter_zst();
    vec_into_iter_rev_zst();
    vec_iter_and_mut_rev();

    assert_eq!(make_vec().capacity(), 4);
    assert_eq!(make_vec_macro(), [1, 2]);
    assert_eq!(make_vec_macro_repeat(), [42; 5]);
    assert_eq!(make_vec_macro_repeat_zeroed(), [0; 7]);

    // Test interesting empty slice comparison
    // (one is a real pointer, one an integer pointer).
    assert_eq!((200..-5).step_by(1).collect::<Vec<isize>>(), []);

    // liballoc has a more extensive test of this, but let's at least do a smoke test here.
    vec_push_ptr_stable();
    vec_extend_ptr_stable();
    vec_truncate_ptr_stable();
    push_str_ptr_stable();

    sort();
    swap();
    swap_remove();
    reverse();
    miri_issue_2759();
}
