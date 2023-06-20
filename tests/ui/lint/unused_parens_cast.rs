// check-pass

#![warn(unused_parens)]

struct Foo(f32);

impl Foo {
    pub fn f32(self) -> f32 {
        64.0f32
    }
}

fn bar() -> f32 {
    3.0f32
}

mod inner {
    pub mod yet_inner {
        pub mod most_inner {
            pub static VERY_LONG_PATH: f32 = 99.0f32;
        }
    }
}

fn basic_test() {
    // should fire
    let one = 1.0f32;
    let _ = (one) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    let _ = (inner::yet_inner::most_inner::VERY_LONG_PATH) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    let _ = (Foo(1.0f32).0) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    let _ = (Foo(1.0f32).f32()) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    let _ = (bar()) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    let baz = [4.0f32];
    let _ = (baz[0]) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    // following is technically unnecessary, but is allowed because it may confusing.
    let _ = (-1.0f32) as f64;
    let x = Box::new(-1.0f32);
    let _ = (*x) as f64;
    //~^ WARN unnecessary parentheses around cast expression
    // cast is left-assoc
    let _ = (true as u8) as u16;
    //~^ WARN unnecessary parentheses around cast expression
    // should not fire
    let _ = (1.0f32 * 2.0f32) as f64;
    let _ = (1.0f32 / 2.0f32) as f64;
    let _ = (1.0f32 % 2.0f32) as f64;
    let _ = (1.0f32 + 2.0f32) as f64;
    let _ = (1.0f32 - 2.0f32) as f64;
    let _ = (42 << 1) as i64;
    let _ = (42 >> 1) as i64;
    let _ = (42 & 0x1F) as f64;
    let _ = (42 ^ 0x1F) as f64;
    let _ = (42 | 0x1F) as f64;
    let _ = (1.0f32 == 2.0f32) as u8;
    let _ = (1.0f32 != 2.0f32) as u8;
    let _ = (1.0f32 < 2.0f32) as u8;
    let _ = (1.0f32 > 2.0f32) as u8;
    let _ = (1.0f32 <= 2.0f32) as u8;
    let _ = (1.0f32 >= 2.0f32) as u8;
    let _ = (true && false) as u8;
    let _ = (true || false) as u8;
    // skipped range: `as`-cast does not allow non-primitive cast
    // also skipped compound operator
}

fn issue_88519() {
    let _ = ({ 1 }) as i64;
    let _ = (match 0 { x => x }) as i64;
    let _ = (if true { 16 } else { 42 }) as i64;
}

fn issue_51185() -> impl Into<for<'a> fn(&'a ())> {
    // removing parens will change semantics, and make compile does not pass
    (|_| {}) as for<'a> fn(&'a ())
}

fn issue_clippy_10557() {
    let x = 0f32;
    let y = 0f32;
    let width = 100f32;
    let height = 100f32;

    new_rect((x) as f64, (y) as f64, (width) as f64, (height) as f64);
    //~^ WARN unnecessary parentheses around cast expression
    //~^^ WARN unnecessary parentheses around cast expression
    //~^^^ WARN unnecessary parentheses around cast expression
    //~^^^^ WARN unnecessary parentheses around cast expression
}

fn new_rect(x: f64, y: f64, width: f64, height: f64) {

}

fn main() {
}
