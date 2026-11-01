// This test makes sure that we don't emit a long list of possible values
// but that we stop at a fix point and say "and X more".
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()
//@ normalize-stderr: "and \d+ more" -> "and X more"
//@ normalize-stderr: "`[a-zA-Z0-9_\.-]+`" -> "`xxx`"

fn main() {
    cfg!(target_feature = "zebra");
    //~^ WARNING unexpected `cfg` condition value
}
