//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

pub fn main() {
    println!("===== WITHOUT THREADS =====");
    test00();
}

fn test00_start(ch: &Sender<isize>, message: isize, count: isize) {
    println!("Starting test00_start");
    let mut i: isize = 0;
    while i < count {
        println!("Sending Message");
        ch.send(message + 0).unwrap();
        i = i + 1;
    }
    println!("Ending test00_start");
}

fn test00() {
    let number_of_tasks: isize = 16;
    let number_of_messages: isize = 4;

    println!("Creating tasks");

    let (tx, rx) = channel();

    let mut i: isize = 0;

    // Create and spawn threads...
    let mut results = Vec::new();
    while i < number_of_tasks {
        let tx = tx.clone();
        results.push(thread::spawn({
            let i = i;
            move || test00_start(&tx, i, number_of_messages)
        }));
        i = i + 1;
    }

    // Read from spawned threads...
    let mut sum = 0;
    for _r in &results {
        i = 0;
        while i < number_of_messages {
            let value = rx.recv().unwrap();
            sum += value;
            i = i + 1;
        }
    }

    // Join spawned threads...
    for r in results {
        r.join();
    }

    println!("Completed: Final number is: ");
    println!("{}", sum);
    // assert (sum == (((number_of_threads * (number_of_threads - 1)) / 2) *
    //       number_of_messages));
    assert_eq!(sum, 480);
}
