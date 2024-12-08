//@error-in-other-file: aborted execution
// Backtraces vary wildly between platforms, we have to normalize away almost the entire thing.
// Full backtraces avoid annoying empty line differences.
//@compile-flags: -Zmiri-backtrace=full
//@normalize-stderr-test: "'main'|'<unnamed>'" -> "$$NAME"
//@normalize-stderr-test: ".*(note|-->|\|).*\n" -> ""

pub struct NoisyDrop {}

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        panic!("ow");
    }
}

thread_local! {
    pub static NOISY: NoisyDrop = const { NoisyDrop {} };
}

fn main() {
    NOISY.with(|_| ());
}
