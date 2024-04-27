// Need to disable preemption to stay on the supported MVP codepath in mio.
//@compile-flags: -Zmiri-permissive-provenance -Zmiri-preemption-rate=0
//@only-target-x86_64-unknown-linux: support for tokio exists only on linux and x86

#[tokio::main]
async fn main() {}
