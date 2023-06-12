//@run-rustfix
#![deny(clippy::explicit_iter_loop)]
#![allow(
    clippy::linkedlist,
    clippy::similar_names,
    clippy::needless_borrow,
    clippy::deref_addrof,
    dead_code
)]

use core::slice;
use std::collections::*;

fn main() {
    let mut vec = vec![1, 2, 3, 4];

    for _ in vec.iter() {}
    for _ in vec.iter_mut() {}

    let rvec = &vec;
    for _ in rvec.iter() {}

    let rmvec = &mut vec;
    for _ in rmvec.iter() {}
    for _ in rmvec.iter_mut() {}

    for _ in &vec {} // these are fine
    for _ in &mut vec {} // these are fine

    for _ in [1, 2, 3].iter() {}

    for _ in (&mut [1, 2, 3]).iter() {}

    for _ in [0; 32].iter() {}
    for _ in [0; 33].iter() {}

    let ll: LinkedList<()> = LinkedList::new();
    for _ in ll.iter() {}
    let rll = &ll;
    for _ in rll.iter() {}

    let vd: VecDeque<()> = VecDeque::new();
    for _ in vd.iter() {}
    let rvd = &vd;
    for _ in rvd.iter() {}

    let bh: BinaryHeap<()> = BinaryHeap::new();
    for _ in bh.iter() {}

    let hm: HashMap<(), ()> = HashMap::new();
    for _ in hm.iter() {}

    let bt: BTreeMap<(), ()> = BTreeMap::new();
    for _ in bt.iter() {}

    let hs: HashSet<()> = HashSet::new();
    for _ in hs.iter() {}

    let bs: BTreeSet<()> = BTreeSet::new();
    for _ in bs.iter() {}

    struct NoIntoIter();
    impl NoIntoIter {
        fn iter(&self) -> slice::Iter<u8> {
            unimplemented!()
        }

        fn iter_mut(&mut self) -> slice::IterMut<u8> {
            unimplemented!()
        }
    }
    let mut x = NoIntoIter();
    for _ in x.iter() {} // no error
    for _ in x.iter_mut() {} // no error

    struct IntoIterDiffTy;
    impl IntoIterator for &'_ IntoIterDiffTy {
        type Item = &'static ();
        type IntoIter = core::slice::Iter<'static, ()>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }
    impl IntoIterDiffTy {
        fn iter(&self) -> core::slice::Iter<'static, i32> {
            unimplemented!()
        }
    }
    let x = IntoIterDiffTy;
    for _ in x.iter() {}

    struct IntoIterDiffSig;
    impl IntoIterator for &'_ IntoIterDiffSig {
        type Item = &'static ();
        type IntoIter = core::slice::Iter<'static, ()>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }
    impl IntoIterDiffSig {
        fn iter(&self, _: u32) -> core::slice::Iter<'static, ()> {
            unimplemented!()
        }
    }
    let x = IntoIterDiffSig;
    for _ in x.iter(0) {}

    struct IntoIterDiffLt<'a>(&'a ());
    impl<'a> IntoIterator for &'a IntoIterDiffLt<'_> {
        type Item = &'a ();
        type IntoIter = core::slice::Iter<'a, ()>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }
    impl<'a> IntoIterDiffLt<'a> {
        fn iter(&self) -> core::slice::Iter<'a, ()> {
            unimplemented!()
        }
    }
    let x = IntoIterDiffLt(&());
    for _ in x.iter() {}

    struct CustomType;
    impl<'a> IntoIterator for &'a CustomType {
        type Item = &'a u32;
        type IntoIter = core::slice::Iter<'a, u32>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }
    impl<'a> IntoIterator for &'a mut CustomType {
        type Item = &'a mut u32;
        type IntoIter = core::slice::IterMut<'a, u32>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }
    impl CustomType {
        fn iter(&self) -> <&'_ Self as IntoIterator>::IntoIter {
            panic!()
        }

        fn iter_mut(&mut self) -> core::slice::IterMut<'_, u32> {
            panic!()
        }
    }
    let mut x = CustomType;
    for _ in x.iter() {}
    for _ in x.iter_mut() {}

    let r = &x;
    for _ in r.iter() {}
}
