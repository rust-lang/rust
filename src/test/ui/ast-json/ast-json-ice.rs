// Test that AST json serialization doesn't ICE (#63728).

// revisions: expand noexpand

//[expand] compile-flags: -Zast-json
//[noexpand] compile-flags: -Zast-json-noexpand

// check-pass
// dont-check-compiler-stdout - don't check for any AST change.

#![feature(asm)]

enum V {
    A(i32),
    B { f: [i64; 3 + 4] }
}

trait X {
    type Output;
    fn read(&self) -> Self::Output;
    fn write(&mut self, _: Self::Output);
}

macro_rules! call_println {
    ($y:ident) => { println!("{}", $y) }
}

fn main() {
    #[cfg(any(target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"))]
    unsafe { asm!(""::::); }

    let x: (i32) = 35;
    let y = x as i64<> + 5;

    call_println!(y);

    struct A;
}
