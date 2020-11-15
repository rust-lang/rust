fn main() {}

struct S { x : u32 }

#[cfg(FALSE)]
fn foo() {
    S { x: 5, .. }; //~ ERROR destructuring assignments are unstable
}
