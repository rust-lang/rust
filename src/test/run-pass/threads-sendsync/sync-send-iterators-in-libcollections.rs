// run-pass

#![allow(warnings)]
#![feature(drain, collections_bound, btree_range)]

use std::collections::BinaryHeap;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::LinkedList;
use std::collections::VecDeque;
use std::collections::HashMap;
use std::collections::HashSet;

use std::mem;
use std::ops::Bound::Included;

fn is_sync<T>(_: T) where T: Sync {}
fn is_send<T>(_: T) where T: Send {}

macro_rules! all_sync_send {
    ($ctor:expr, $($iter:ident),+) => ({
        $(
            let mut x = $ctor;
            is_sync(x.$iter());
            let mut y = $ctor;
            is_send(y.$iter());
        )+
    })
}

macro_rules! is_sync_send {
    ($ctor:expr, $iter:ident($($param:expr),+)) => ({
        let mut x = $ctor;
        is_sync(x.$iter($( $param ),+));
        let mut y = $ctor;
        is_send(y.$iter($( $param ),+));
    })
}

fn main() {
    // The iterator "generator" list should exhaust what corresponding
    // implementations have where `Sync` and `Send` semantics apply.
    all_sync_send!(BinaryHeap::<usize>::new(), iter, drain, into_iter);

    all_sync_send!(BTreeMap::<usize, usize>::new(), iter, iter_mut, into_iter, keys, values);
    is_sync_send!(BTreeMap::<usize, usize>::new(), range((Included(&0), Included(&9))));
    is_sync_send!(BTreeMap::<usize, usize>::new(), range_mut((Included(&0), Included(&9))));

    all_sync_send!(BTreeSet::<usize>::new(), iter, into_iter);
    is_sync_send!(BTreeSet::<usize>::new(), range((Included(&0), Included(&9))));
    is_sync_send!(BTreeSet::<usize>::new(), difference(&BTreeSet::<usize>::new()));
    is_sync_send!(BTreeSet::<usize>::new(), symmetric_difference(&BTreeSet::<usize>::new()));
    is_sync_send!(BTreeSet::<usize>::new(), intersection(&BTreeSet::<usize>::new()));
    is_sync_send!(BTreeSet::<usize>::new(), union(&BTreeSet::<usize>::new()));

    all_sync_send!(HashMap::<usize, usize>::new(), iter, iter_mut, drain, into_iter, keys, values);
    is_sync_send!(HashMap::<usize, usize>::new(), entry(0));
    all_sync_send!(HashSet::<usize>::new(), iter, drain, into_iter);
    is_sync_send!(HashSet::<usize>::new(), difference(&HashSet::<usize>::new()));
    is_sync_send!(HashSet::<usize>::new(), symmetric_difference(&HashSet::<usize>::new()));
    is_sync_send!(HashSet::<usize>::new(), intersection(&HashSet::<usize>::new()));
    is_sync_send!(HashSet::<usize>::new(), union(&HashSet::<usize>::new()));

    all_sync_send!(LinkedList::<usize>::new(), iter, iter_mut, into_iter);

    all_sync_send!(VecDeque::<usize>::new(), iter, iter_mut, into_iter);
    is_sync_send!(VecDeque::<usize>::new(), drain(..));

    all_sync_send!(Vec::<usize>::new(), into_iter);
    is_sync_send!(Vec::<usize>::new(), drain(..));
    is_sync_send!(String::new(), drain(..));
}
