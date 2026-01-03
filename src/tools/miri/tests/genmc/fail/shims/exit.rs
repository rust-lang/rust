//@ compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

fn main() {
    std::thread::spawn(|| {
        unsafe { std::hint::unreachable_unchecked() }; //~ERROR: entering unreachable code
    });
    // If we exit immediately, we might entirely miss the UB in the other thread.
    std::process::exit(0);
}
