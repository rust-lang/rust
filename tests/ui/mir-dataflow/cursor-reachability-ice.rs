//@ build-pass
//@ compile-flags: -Zmir-enable-passes=-SimplifyCfg-initial,-SimplifyCfg-promote-consts,-SimplifyCfg-post-analysis

// Regression test for issue #160945
// The goal of this test is to verify if, with the specified SimplifyCfg passes disabled, rustc can query an unreachable MIR block
// containing a drop without applying its effects.
// Previously, querying dataflow state at this unreachable block caused ResultsCursor to ICE.

fn ice<T>(_place: T) {
    panic!()
}

fn main() {
    ice(());
}
