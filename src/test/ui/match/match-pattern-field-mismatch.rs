fn main() {
    enum color {
        rgb(usize, usize, usize),
        cmyk(usize, usize, usize, usize),
        no_color,
    }

    fn foo(c: color) {
        match c {
          color::rgb(_, _) => { }
          //~^ ERROR this pattern has 2 fields, but the corresponding tuple variant has 3 fields
          color::cmyk(_, _, _, _) => { }
          color::no_color => { }
        }
    }
}
