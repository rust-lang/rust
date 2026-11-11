//! This reproduces #4655 every single time
//@ compile-flags: -Zmiri-fixed-schedule -Zmiri-disable-stacked-borrows
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{ptr, thread};

const SIZE: usize = 256;

static mut ARRAY: [u8; SIZE] = [0; _];
// Everything strictly less than this has been initialized by the sender.
static POS: AtomicUsize = AtomicUsize::new(0);

fn main() {
    // Sender
    let th = std::thread::spawn(|| {
        for i in 0..SIZE {
            unsafe { ptr::write(&raw mut ARRAY[i], 1) };
            POS.store(i + 1, Ordering::Release);

            thread::yield_now();

            // We are the only writer, so we can do non-atomic reads as well.
            unsafe { (&raw const POS).cast::<usize>().read() };
        }
    });

    // Receiver
    loop {
        let i = POS.load(Ordering::Acquire);

        if i > 0 {
            unsafe { ptr::read(&raw const ARRAY[i - 1]) };
        }

        if i == SIZE {
            break;
        }

        thread::yield_now();
    }

    th.join().unwrap();
}
