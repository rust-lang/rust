//@ run-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::sync::Arc;

fn foo() {
    let x = Arc::new("foo".to_owned());
    let y = Arc::new(vec![1, 2, 3]);
    let z = Arc::new(1);

    for _ in 0..10 {
        let handler = std::thread::spawn(|| {
            drop((x.use, y.use, z.use));
        });
        handler.join().unwrap();
    }
}

fn bar() {
    use std::sync::Arc;
    use std::thread;

    let five = Arc::new(5);

    for _ in 0..10 {
        let handler = thread::spawn(|| {
            println!("{:?}", five.use);
        });
        handler.join().unwrap();
    }
}

fn main() {
    foo();
    bar();
}
