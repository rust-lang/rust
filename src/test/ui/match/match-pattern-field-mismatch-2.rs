fn main() {
    enum color {
        rgb(usize, usize, usize),
        cmyk(usize, usize, usize, usize),
        no_color,
    }

    fn foo(c: color) {
        match c {
          color::rgb(_, _, _) => { }
          color::cmyk(_, _, _, _) => { }
          color::no_color(_) => { }
          //~^ ERROR expected tuple struct/variant, found unit variant `color::no_color`
        }
    }
}
