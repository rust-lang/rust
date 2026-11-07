//@ revisions: edition2018 edition2024
//@ [edition2018] edition: 2018
//@ [edition2024] edition: 2024
//@ [edition2024] check-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::sync::Arc;

fn foo() {
    // The type is a tuple and doesn't implement UseCloned
    let x = (Arc::new("foo".to_owned()), Arc::new(vec![1, 2, 3]), Arc::new(1));
    for _ in 0..10 {
        let handler = std::thread::spawn(use || {
            //[edition2018]~^ ERROR use of moved value: `x` [E0382]
            drop((x.0, x.1, x.2));
        });
        handler.join().unwrap();
    }
}

fn bar() {
    let x = Arc::new("foo".to_owned());
    let y = Arc::new(vec![1, 2, 3]);
    let z = Arc::new(1);

    for _ in 0..10 {
        let handler = std::thread::spawn(use || {
            drop((x, y, z));
        });
        handler.join().unwrap();
    }
}

fn baz() {
    use std::sync::Arc;
    use std::thread;

    let five = Arc::new(5);

    for _ in 0..10 {
        let handler = thread::spawn(use || {
            println!("{five:?}");
        });
        handler.join().unwrap();
    }
}

fn main() {}
