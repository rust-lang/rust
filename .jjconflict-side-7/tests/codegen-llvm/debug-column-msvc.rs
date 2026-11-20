// Verify that no column information is emitted for MSVC targets
//
//@ only-msvc
//@ compile-flags: -C debuginfo=2

// CHECK-NOT: !DILexicalBlock({{.*}}column: {{.*}})
// CHECK-NOT: !DILocation({{.*}}column: {{.*}})

pub fn add(a: u32, b: u32) -> u32 {
    a + b
}

fn main() {
    let c = add(1, 2);
    println!("{}", c);
}
