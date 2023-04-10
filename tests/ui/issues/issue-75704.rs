// Caused an infinite loop during SimplifyCfg MIR transform previously.
//
// build-pass

fn main() {
    loop { continue; }
}
