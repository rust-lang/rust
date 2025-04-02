//@ revisions: empty unprefixed all_unknown all_known mixed

//@[empty] compile-flags: -Zmir-enable-passes=
//@[empty] error-pattern error: incorrect value `` for unstable option `mir-enable-passes` - a comma-separated list of strings, with elements beginning with + or - was expected

//@[unprefixed] compile-flags: -Zmir-enable-passes=CheckAlignment
//@[unprefixed] error-pattern error: incorrect value `CheckAlignment` for unstable option `mir-enable-passes` - a comma-separated list of strings, with elements beginning with + or - was expected

//@[all_unknown] check-pass
//@[all_unknown] compile-flags: -Zmir-enable-passes=+ThisPass,-DoesNotExist
//@[all_unknown] error-pattern: warning: MIR pass `ThisPass` is unknown and will be ignored
//@[all_unknown] error-pattern: warning: MIR pass `DoesNotExist` is unknown and will be ignored

//@[all_known] check-pass
//@[all_known] compile-flags: -Zmir-enable-passes=+CheckAlignment,+LowerIntrinsics

//@[mixed] check-pass
//@[mixed] compile-flags: -Zmir-enable-passes=+ThisPassDoesNotExist,+CheckAlignment
//@[mixed] error-pattern: warning: MIR pass `ThisPassDoesNotExist` is unknown and will be ignored

fn main() {}

//[empty]~? ERROR incorrect value `` for unstable option `mir-enable-passes`
//[unprefixed]~? ERROR incorrect value `CheckAlignment` for unstable option `mir-enable-passes`
