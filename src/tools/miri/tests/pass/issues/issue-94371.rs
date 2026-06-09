#[repr(C)]
struct Demo(u64, bool, u64, u32, u64, u64, u64);

fn test() -> (Demo, Demo) {
    let mut x = Demo(1, true, 3, 4, 5, 6, 7);
    let mut y = Demo(10, false, 12, 13, 14, 15, 16);
    std::mem::swap(&mut x, &mut y);
    (x, y)
}

fn main() {
    drop(test());
}
