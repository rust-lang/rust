//@ edition:2021
//@ run-pass

#[derive(Debug, PartialEq, Eq)]
pub enum Color {
    RGB(u8, u8, u8),
}

fn main() {
    let mut color = Color::RGB(0, 0, 0);
    let mut red = |v| {
        let Color::RGB(ref mut r, _, _) = color;
        *r = v;
    };
    let mut green = |v| {
        let Color::RGB(_, ref mut g, _) = color;
        *g = v;
    };
    let mut blue = |v| {
        let Color::RGB(_, _, ref mut b) = color;
        *b = v;
    };
    red(1);
    green(2);
    blue(3);
    assert_eq!(Color::RGB(1, 2, 3), color);
}
