#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

#[lang="copy"]
trait Copy {}

#[lang="freeze"]
trait Freeze {}

#[lang="mul"]
trait Mul<RHS = Self> {
    type Output;

    #[must_use]
    fn mul(self, rhs: RHS) -> Self::Output;
}

impl Mul for u8 {
    type Output = u8;

    fn mul(self, rhs: u8) -> u8 {
        self * rhs
    }
}

#[lang="panic"]
fn panic(_expr_file_line_col: &(&'static str, &'static str, u32, u32)) -> ! {
    loop {}
}

fn abc(a: u8) -> u8 {
    a * 2
}

/*fn bcd(b: bool, a: u8) -> u8 {
    if b {
        a * 2
    } else {
        a * 3
    }
}*/

fn call() {
    abc(42);
}
