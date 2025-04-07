//@ revisions: empty unprefixed all_unknown all_known mixed

//@[empty] compile-flags: -Zmir-enable-passes=

//@[unprefixed] compile-flags: -Zmir-enable-passes=CheckAlignment

//@[all_unknown] check-pass
//@[all_unknown] compile-flags: -Zmir-enable-passes=+ThisPass,-DoesNotExist

//@[all_known] check-pass
//@[all_known] compile-flags: -Zmir-enable-passes=+CheckAlignment,+LowerIntrinsics

//@[mixed] check-pass
//@[mixed] compile-flags: -Zmir-enable-passes=+ThisPassDoesNotExist,+CheckAlignment

fn main() {}

//[empty]~? ERROR incorrect value `` for unstable option `mir-enable-passes`
//[unprefixed]~? ERROR incorrect value `CheckAlignment` for unstable option `mir-enable-passes`
//[mixed]~? WARN MIR pass `ThisPassDoesNotExist` is unknown and will be ignored
//[mixed]~? WARN MIR pass `ThisPassDoesNotExist` is unknown and will be ignored
//[all_unknown]~? MIR pass `ThisPass` is unknown and will be ignored
//[all_unknown]~? MIR pass `DoesNotExist` is unknown and will be ignored
//[all_unknown]~? MIR pass `ThisPass` is unknown and will be ignored
//[all_unknown]~? MIR pass `DoesNotExist` is unknown and will be ignored
