#![warn(clippy::unsound_collection_transmute)]

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::mem::transmute;

fn main() {
    unsafe {
        // wrong size
        let _ = transmute::<_, Vec<u32>>(vec![0u8]);
        // wrong layout
        let _ = transmute::<_, Vec<[u8; 4]>>(vec![1234u32]);

        // wrong size
        let _ = transmute::<_, VecDeque<u32>>(VecDeque::<u8>::new());
        // wrong layout
        let _ = transmute::<_, VecDeque<u32>>(VecDeque::<[u8; 4]>::new());

        // wrong size
        let _ = transmute::<_, BinaryHeap<u32>>(BinaryHeap::<u8>::new());
        // wrong layout
        let _ = transmute::<_, BinaryHeap<u32>>(BinaryHeap::<[u8; 4]>::new());

        // wrong size
        let _ = transmute::<_, BTreeSet<u32>>(BTreeSet::<u8>::new());
        // wrong layout
        let _ = transmute::<_, BTreeSet<u32>>(BTreeSet::<[u8; 4]>::new());

        // wrong size
        let _ = transmute::<_, HashSet<u32>>(HashSet::<u8>::new());
        // wrong layout
        let _ = transmute::<_, HashSet<u32>>(HashSet::<[u8; 4]>::new());

        // wrong size
        let _ = transmute::<_, BTreeMap<u8, u32>>(BTreeMap::<u8, u8>::new());
        let _ = transmute::<_, BTreeMap<u8, u32>>(BTreeMap::<u32, u32>::new());
        // wrong layout
        let _ = transmute::<_, BTreeMap<u8, u32>>(BTreeMap::<u8, [u8; 4]>::new());
        let _ = transmute::<_, BTreeMap<u32, u32>>(BTreeMap::<[u8; 4], u32>::new());

        // wrong size
        let _ = transmute::<_, HashMap<u8, u32>>(HashMap::<u8, u8>::new());
        let _ = transmute::<_, HashMap<u8, u32>>(HashMap::<u32, u32>::new());
        // wrong layout
        let _ = transmute::<_, HashMap<u8, u32>>(HashMap::<u8, [u8; 4]>::new());
        let _ = transmute::<_, HashMap<u32, u32>>(HashMap::<[u8; 4], u32>::new());
    }
}
