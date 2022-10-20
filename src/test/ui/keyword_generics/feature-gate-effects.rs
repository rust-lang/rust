// This is a fake feature gate test. There is currently no way
// to fail a test only with the feature gate missing.
// FIXME(effects): make this an actual test once effects enable more code to compile

fn main() {
    "just fail this test" //~ ERROR mismatched types
}
