//@ revisions: dflt ncyr ycnr ncnr ycyr
//@ add-minicore
//@ needs-llvm-components: systemz
//@ compile-flags: -Zinstrument-function=fentry -Copt-level=0 --target=s390x-unknown-linux-gnu
//@[ncyr] compile-flags: -Zinstrument-fentry-opts=no-call,record
//@[ncnr] compile-flags: -Zinstrument-fentry-opts=no-call,no-record
//@[ycnr] compile-flags: -Zinstrument-fentry-opts=call,no-record
//@[ycyr] compile-flags: -Zinstrument-fentry-opts=call,record
#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

// dflt: attributes #{{.*}} {{.*}} "fentry-call"="true"
// dflt-NOT: attributes #{{.*}} {{.*}} "mnop-mcount"
// dflt-NOT: attributes #{{.*}} {{.*}} "mrecord-mcount"
//
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
