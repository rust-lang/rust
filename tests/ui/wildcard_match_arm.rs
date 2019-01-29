#![deny(clippy::wildcard_match_arm)]

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Color {
    Red,
    Green,
    Blue,
    Rgb(u8, u8, u8),
    Cyan,
}

impl Color {
    fn is_monochrome(self) -> bool {
        match self {
            Color::Red | Color::Green | Color::Blue => true,
            Color::Rgb(r, g, b) => r | g == 0 || r | b == 0 || g | b == 0,
            Color::Cyan => false,
        }
    }
}

fn main() {
    let color = Color::Rgb(0, 0, 127);
    match color {
        Color::Red => println!("Red"),
        _ => eprintln!("Not red"),
    };
    match color {
        Color::Red => {},
        Color::Green => {},
        Color::Blue => {},
        Color::Cyan => {},
        c if c.is_monochrome() => {},
        Color::Rgb(_, _, _) => {},
    };
}
