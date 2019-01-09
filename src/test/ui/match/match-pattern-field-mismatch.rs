fn main() {
    enum Color {
        Rgb(usize, usize, usize),
        Cmyk(usize, usize, usize, usize),
        NoColor,
    }

    fn foo(c: Color) {
        match c {
          Color::Rgb(_, _) => { }
          //~^ ERROR this pattern has 2 fields, but the corresponding tuple variant has 3 fields
          Color::Cmyk(_, _, _, _) => { }
          Color::NoColor => { }
        }
    }
}
