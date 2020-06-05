// checks case typos with libstd::collections structs
fn main(){}

fn test_btm(_x: BtreeMap<(), ()>){}
//~^ ERROR: cannot find type `BtreeMap` in this scope
fn test_bts(_x: BtreeSet<()>){}
//~^ ERROR: cannot find type `BtreeSet` in this scope
fn test_binh(_x: Binaryheap<()>){}
//~^ ERROR: cannot find type `Binaryheap` in this scope
fn test_hashm(_x: Hashmap<String, ()>){}
//~^ ERROR: cannot find type `Hashmap` in this scope
fn test_hashs(_x: Hashset<()>){}
//~^ ERROR: cannot find type `Hashset` in this scope
fn test_llist(_x: Linkedlist<()>){}
//~^ ERROR: cannot find type `Linkedlist` in this scope
fn test_vd(_x: Vecdeque<()>){}
//~^ ERROR: cannot find type `Vecdeque` in this scope
