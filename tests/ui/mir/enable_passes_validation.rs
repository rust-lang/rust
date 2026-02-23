//@ revisions: empty unprefixed all_unknown all_known mixed
//@ revisions: enum_not_in_pass_names enum_in_pass_names

//@[empty] compile-flags: -Zmir-enable-passes=

//@[unprefixed] compile-flags: -Zmir-enable-passes=CheckAlignment

//@[all_unknown] check-pass
//@[all_unknown] compile-flags: -Zmir-enable-passes=+ThisPass,-DoesNotExist

//@[all_known] check-pass
//@[all_known] compile-flags: -Zmir-enable-passes=+CheckAlignment,+LowerIntrinsics

//@[mixed] check-pass
//@[mixed] compile-flags: -Zmir-enable-passes=+ThisPassDoesNotExist,+CheckAlignment

//@[enum_not_in_pass_names] check-pass
//@[enum_not_in_pass_names] compile-flags: -Zmir-enable-passes=+SimplifyCfg

//@[enum_in_pass_names] check-pass
//@[enum_in_pass_names] compile-flags: -Zmir-enable-passes=+AddCallGuards

fn main() {}

//[empty]~? ERROR incorrect value `` for unstable option `mir-enable-passes`
//[unprefixed]~? ERROR incorrect value `CheckAlignment` for unstable option `mir-enable-passes`
//[mixed]~? WARN MIR pass `ThisPassDoesNotExist` is unknown and will be ignored
//[mixed]~? WARN MIR pass `ThisPassDoesNotExist` is unknown and will be ignored
//[all_unknown]~? WARN MIR pass `ThisPass` is unknown and will be ignored
//[all_unknown]~? WARN MIR pass `DoesNotExist` is unknown and will be ignored
//[all_unknown]~? WARN MIR pass `ThisPass` is unknown and will be ignored
//[all_unknown]~? WARN MIR pass `DoesNotExist` is unknown and will be ignored
//[enum_not_in_pass_names]~? WARN MIR pass `SimplifyCfg` is unknown and will be ignored
//[enum_not_in_pass_names]~? WARN MIR pass `SimplifyCfg` is unknown and will be ignored
