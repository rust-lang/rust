// compile-flags: -O
#![crate_type = "lib"]

// CHECK: @func2 = {{.*}}alias{{.*}}@func1

#[no_mangle]
pub fn func1(c: char) -> bool {
    c == 's' || c == 'm' || c == 'h' || c == 'd' || c == 'w'
}

#[no_mangle]
pub fn func2(c: char) -> bool {
    matches!(c, 's' | 'm' | 'h' | 'd' | 'w')
}
