#![feature(unboxed_closures, fn_traits)]

struct S3 {
    x: i32,
    y: i32,
}

impl FnOnce<(i32, i32)> for S3 {
    type Output = i32;
    extern "rust-call" fn call_once(self, (z, zz): (i32, i32)) -> i32 {
        self.x * self.y * z * zz
    }
}

fn main() {
    let s = S3 { x: 3, y: 3 };
    let ans = s(3, 1);
    assert_eq!(ans, 27);
}
