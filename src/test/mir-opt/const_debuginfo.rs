// compile-flags: -C overflow-checks=no -Zunsound-mir-opts

struct Point {
    x: u32,
    y: u32,
}

fn main() {
    let x = 1u8;
    let y = 2u8;
    let z = 3u8;
    let sum = x + y + z;

    let s = "hello, world!";

    let f = (true, false, 123u32);

    let o = Some(99u16);

    let p = Point { x: 32, y: 32 };
    let a = p.x + p.y;
}

// EMIT_MIR const_debuginfo.main.ConstDebugInfo.diff
