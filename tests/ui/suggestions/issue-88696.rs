// This test case should ensure that miniz_oxide isn't
// suggested, since it's not a direct dependency.

fn a() -> Result<u64, i32> {
    Err(1)
}

fn b() -> Result<u32, i32> {
    a().into() //~ERROR [E0277]
}

fn main() {
    let _ = dbg!(b());
}
