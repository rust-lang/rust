//@ compile-flags: -g -Zmir-opt-level=0

#![crate_type = "lib"]

#[inline(never)]
fn swizzle(a: u32, b: u32, c: u32) -> (u32, (u32, u32)) {
    (b, (c, a))
}

pub fn work() {
    let mut a = 1;
    let mut b = 2;
    let mut c = 3;
    (a, (b, c)) = swizzle(a, b, c);
    println!("{a} {b} {c}");
}

// CHECK-NOT: !DILocalVariable(name: "lhs",
