// no-system-llvm
// compile-flags: -Coverflow-checks=no -O
// revisions: YES NO
// [YES]compile-flags: -Zfewer-names=yes
// [NO] compile-flags: -Zfewer-names=no
#![crate_type = "lib"]

#[no_mangle]
pub fn sum(x: u32, y: u32) -> u32 {
// YES-LABEL: define{{.*}}i32 @sum(i32 noundef %0, i32 noundef %1)
// YES-NEXT:    %3 = add i32 %1, %0
// YES-NEXT:    ret i32 %3

// NO-LABEL: define{{.*}}i32 @sum(i32 noundef %x, i32 noundef %y)
// NO-NEXT:  start:
// NO-NEXT:    %0 = add i32 %y, %x
// NO-NEXT:    ret i32 %0
    let z = x + y;
    z
}
