// This test prints all the possible values for the `target_feature` cfg
// as a way to assert the expected values and reflect on any changes made
// to the `target_feature` cfg in the compiler.
//
// The output of this test  does not reflect the actual output seen by
// users which will see a truncated list of possible values (at worst).
//
// In case of test output differences, just `--bless` the test.
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg() -Zcheck-cfg-all-expected
//@ normalize-stderr: "`, `" -> "`\n`"

fn main() {
    cfg!(target_feature = "_UNEXPECTED_VALUE");
    //~^ WARNING unexpected `cfg` condition value
}
