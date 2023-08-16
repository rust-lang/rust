// Checks that `-Z instrument-xray` does not allow duplicates.
//
// needs-xray
//@compile-flags: -Z instrument-xray=always,always
//@error-in-other-file: incorrect value `always,always` for unstable option `instrument-xray`

fn main() {}
