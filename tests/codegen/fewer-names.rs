//@ compile-flags: -Coverflow-checks=no -O
//@ revisions: yes no
//@ [yes]compile-flags: -Zfewer-names=yes
//@ [no] compile-flags: -Zfewer-names=no
#![crate_type = "lib"]

#[no_mangle]
pub fn sum(x: u32, y: u32) -> u32 {
    // CHECK-YES-LABEL: define{{.*}}i32 @sum(i32 noundef %0, i32 noundef %1)
    // CHECK-YES-NEXT:    %3 = add i32 %1, %0
    // CHECK-YES-NEXT:    ret i32 %3

    // CHECK-NO-LABEL: define{{.*}}i32 @sum(i32 noundef %x, i32 noundef %y)
    // CHECK-NO-NEXT:  start:
    // CHECK-NO-NEXT:    %z = add i32 %y, %x
    // CHECK-NO-NEXT:    ret i32 %z
    let z = x + y;
    z
}
