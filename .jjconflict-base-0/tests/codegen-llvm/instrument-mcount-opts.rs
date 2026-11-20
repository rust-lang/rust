//@ revisions: ncyr ycyr ycnr
//@ add-minicore
//@ needs-llvm-components: systemz
//@ compile-flags: -Copt-level=0 --target=s390x-unknown-linux-gnu
//@[ncyr] compile-flags: -Zinstrument-mcount=fentry-nop-record
//@[ycyr] compile-flags: -Zinstrument-mcount=fentry-record
//@[ycnr] compile-flags: -Zinstrument-mcount=fentry
#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

// ncyr: attributes #{{.*}} {{.*}} "fentry-call"="true" "mnop-mcount" "mrecord-mcount"
//
// ncnr: attributes #{{.*}} {{.*}} "fentry-call"="true" "mnop-mcount"
// ncnr-NOT: attributes #{{.*}} {{.*}} "mrecord-mcount"
//
// ycnr: attributes #{{.*}} {{.*}} "fentry-call"="true"
// ycnr-NOT: attributes #{{.*}} {{.*}} "mnop-mcount"
// ycnr-NOT: attributes #{{.*}} {{.*}} "mrecord-mcount"
//
// ycyr: attributes #{{.*}} {{.*}} "fentry-call"="true" "mrecord-mcount"
// ycyr-NOT: attributes #{{.*}} {{.*}} "mnop-mcount"
pub fn foo() {}
