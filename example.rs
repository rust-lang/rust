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

impl Mul for u32 {
    type Output = u32;

    fn mul(self, rhs: u32) -> u32 {
        self * rhs
    }
}

#[lang="panic"]
fn panic(_expr_file_line_col: &(&'static str, &'static str, u32, u32)) -> ! {
    loop {}
}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}

fn abc(a: u32) -> u32 {
    a * 2
}

fn bcd(b: bool, a: u32) -> u32 {
    if b {
        a * 2
    } else {
        a * 3
    }
}

// FIXME make calls work
/*fn call() {
    abc(42);
}

fn indirect_call() {
    let f: fn() = call;
    f();
}*/

enum BoolOption {
    Some(bool),
    None,
}

fn option_unwrap_or(o: BoolOption, d: bool) -> bool {
    match o {
        BoolOption::Some(b) => b,
        BoolOption::None => d,
    }
}

fn ret_42() -> u32 {
    42
}
