// compile-flags: -Zfuture-incompat-test
// check-pass

// The `-Zfuture-incompat-test flag causes any normal warning to be included
// in the future-incompatible report. The stderr output here should mention
// the future incompatible report (as extracted by compiletest).

fn main() {
    let x = 1;
}
