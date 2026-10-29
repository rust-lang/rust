//@ run-crash
//@ revisions: single batch

#![feature(arc_n)]

use std::sync::Arc;

fn main() {
    std::panic::set_hook(Box::new(|_| std::process::exit(0)));

    let original = Arc::new(());
    let _reserved = Arc::clone_n(&original, isize::MAX as usize - 1);

    #[cfg(single)]
    let n = 1;
    #[cfg(batch)]
    let n = 1usize << (usize::BITS / 2);

    let _ = Arc::clone_n(&original, n);
}
