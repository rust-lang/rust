//@error-in-other-file: aborted execution

pub struct NoisyDrop {}

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        panic!("ow");
    }
}

thread_local! {
    pub static NOISY: NoisyDrop = NoisyDrop {};
}

fn main() {
    NOISY.with(|_| ());
}
