use std::collections::BinaryHeap;

fn main() {
    let mut heap: BinaryHeap<i32> = BinaryHeap::new();
    let borrow = heap.peek_mut();

    match (borrow, ()) {
        (Some(_), ()) => {
            println!("{:?}", heap); //~ ERROR cannot borrow `heap` as immutable
        }
        _ => {}
    };
}
