// error-pattern: borrow of moved value

use std::sync::Arc;
use std::thread;

fn main() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arc_v = Arc::new(v);

    thread::spawn(move|| {
        assert_eq!((*arc_v)[3], 4);
    });

    assert_eq!((*arc_v)[2], 3);

    println!("{:?}", *arc_v);
}
