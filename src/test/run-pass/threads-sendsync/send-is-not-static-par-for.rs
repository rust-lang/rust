// run-pass
#![allow(unused_imports)]
use std::thread;
use std::sync::Mutex;

fn par_for<I, F>(iter: I, f: F)
    where I: Iterator,
          I::Item: Send,
          F: Fn(I::Item) + Sync
{
    for item in iter {
        f(item)
    }
}

fn sum(x: &[i32]) {
    let sum_lengths = Mutex::new(0);
    par_for(x.windows(4), |x| {
        *sum_lengths.lock().unwrap() += x.len()
    });

    assert_eq!(*sum_lengths.lock().unwrap(), (x.len() - 3) * 4);
}

fn main() {
    let mut elements = [0; 20];

    // iterators over references into this stack frame
    par_for(elements.iter_mut().enumerate(), |(i, x)| {
        *x = i as i32
    });

    sum(&elements)
}
