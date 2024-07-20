//@ revisions: opt-level-2 opt-level-s
//@[opt-level-s] compile-flags: -Copt-level=s
//@[opt-level-2] compile-flags: -O
#![crate_type = "lib"]

// CHECK: @func{{2|1}} = {{.*}}alias{{.*}}@func{{1|2}}

#[no_mangle]
pub fn func1(c: char) -> bool {
    c == 's' || c == 'm' || c == 'h' || c == 'd' || c == 'w'
}

#[no_mangle]
pub fn func2(c: char) -> bool {
    matches!(c, 's' | 'm' | 'h' | 'd' | 'w')
}
