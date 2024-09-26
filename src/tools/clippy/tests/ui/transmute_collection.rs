#![warn(clippy::unsound_collection_transmute)]
#![allow(clippy::missing_transmute_annotations)]

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::mem::{MaybeUninit, transmute};

fn main() {
    unsafe {
        // wrong size
        let _ = transmute::<_, Vec<u32>>(vec![0u8]);
        //~^ ERROR: transmute from `std::vec::Vec<u8>` to `std::vec::Vec<u32>` with mismat
        //~| NOTE: `-D clippy::unsound-collection-transmute` implied by `-D warnings`
        // wrong layout
        let _ = transmute::<_, Vec<[u8; 4]>>(vec![1234u32]);
        //~^ ERROR: transmute from `std::vec::Vec<u32>` to `std::vec::Vec<[u8; 4]>` with m

        // wrong size
        let _ = transmute::<_, VecDeque<u32>>(VecDeque::<u8>::new());
        //~^ ERROR: transmute from `std::collections::VecDeque<u8>` to `std::collections::
        // wrong layout
        let _ = transmute::<_, VecDeque<u32>>(VecDeque::<[u8; 4]>::new());
        //~^ ERROR: transmute from `std::collections::VecDeque<[u8; 4]>` to `std::collecti

        // wrong size
        let _ = transmute::<_, BinaryHeap<u32>>(BinaryHeap::<u8>::new());
        //~^ ERROR: transmute from `std::collections::BinaryHeap<u8>` to `std::collections
        // wrong layout
        let _ = transmute::<_, BinaryHeap<u32>>(BinaryHeap::<[u8; 4]>::new());
        //~^ ERROR: transmute from `std::collections::BinaryHeap<[u8; 4]>` to `std::collec

        // wrong size
        let _ = transmute::<_, BTreeSet<u32>>(BTreeSet::<u8>::new());
        //~^ ERROR: transmute from `std::collections::BTreeSet<u8>` to `std::collections::
        // wrong layout
        let _ = transmute::<_, BTreeSet<u32>>(BTreeSet::<[u8; 4]>::new());
        //~^ ERROR: transmute from `std::collections::BTreeSet<[u8; 4]>` to `std::collecti

        // wrong size
        let _ = transmute::<_, HashSet<u32>>(HashSet::<u8>::new());
        //~^ ERROR: transmute from `std::collections::HashSet<u8>` to `std::collections::H
        // wrong layout
        let _ = transmute::<_, HashSet<u32>>(HashSet::<[u8; 4]>::new());
        //~^ ERROR: transmute from `std::collections::HashSet<[u8; 4]>` to `std::collectio

        // wrong size
        let _ = transmute::<_, BTreeMap<u8, u32>>(BTreeMap::<u8, u8>::new());
        //~^ ERROR: transmute from `std::collections::BTreeMap<u8, u8>` to `std::collectio
        let _ = transmute::<_, BTreeMap<u8, u32>>(BTreeMap::<u32, u32>::new());
        //~^ ERROR: transmute from `std::collections::BTreeMap<u32, u32>` to `std::collect
        // wrong layout
        let _ = transmute::<_, BTreeMap<u8, u32>>(BTreeMap::<u8, [u8; 4]>::new());
        //~^ ERROR: transmute from `std::collections::BTreeMap<u8, [u8; 4]>` to `std::coll
        let _ = transmute::<_, BTreeMap<u32, u32>>(BTreeMap::<[u8; 4], u32>::new());
        //~^ ERROR: transmute from `std::collections::BTreeMap<[u8; 4], u32>` to `std::col

        // wrong size
        let _ = transmute::<_, HashMap<u8, u32>>(HashMap::<u8, u8>::new());
        //~^ ERROR: transmute from `std::collections::HashMap<u8, u8>` to `std::collection
        let _ = transmute::<_, HashMap<u8, u32>>(HashMap::<u32, u32>::new());
        //~^ ERROR: transmute from `std::collections::HashMap<u32, u32>` to `std::collecti
        // wrong layout
        let _ = transmute::<_, HashMap<u8, u32>>(HashMap::<u8, [u8; 4]>::new());
        //~^ ERROR: transmute from `std::collections::HashMap<u8, [u8; 4]>` to `std::colle
        let _ = transmute::<_, HashMap<u32, u32>>(HashMap::<[u8; 4], u32>::new());
        //~^ ERROR: transmute from `std::collections::HashMap<[u8; 4], u32>` to `std::coll

        let _ = transmute::<_, Vec<u8>>(Vec::<MaybeUninit<u8>>::new());
        let _ = transmute::<_, Vec<*mut u32>>(Vec::<Box<u32>>::new());
    }
}
