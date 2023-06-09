// compile-flags: -C debug-assertions=yes

// EMIT_MIR shifts.shift_signed.built.after.mir
fn shift_signed(small: i8, big: u128, a: i8, b: i32, c: i128) -> ([i8; 3], [u128; 3]) {
    (
        [small >> a, small >> b, small >> c],
        [big << a, big << b, big << c],
    )
}

// EMIT_MIR shifts.shift_unsigned.built.after.mir
fn shift_unsigned(small: u8, big: i128, a: u8, b: u32, c: u128) -> ([u8; 3], [i128; 3]) {
    (
        [small >> a, small >> b, small >> c],
        [big << a, big << b, big << c],
    )
}

fn main() {
}
