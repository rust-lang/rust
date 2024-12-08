//@ run-rustfix
#![allow(unused_variables)]

fn main() {
    let items = vec![1, 2, 3];
    let ref_items: &[i32] = &items;
    let items_clone: Vec<i32> = ref_items.clone();
    //~^ ERROR mismatched types

    // in that case no suggestion will be triggered
    let items_clone_2: Vec<i32> = items.clone();

    let s = "hi";
    let string: String = s.clone();
    //~^ ERROR mismatched types

    // in that case no suggestion will be triggered
    let s2 = "hi";
    let string_2: String = s2.to_string();
}
