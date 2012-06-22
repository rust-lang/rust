fn main() {
    enum color {
        rgb(uint, uint, uint),
        cmyk(uint, uint, uint, uint),
        no_color,
    }

    fn foo(c: color) {
        alt c {
          rgb(_, _) { }
          //!^ ERROR this pattern has 2 fields, but the corresponding variant has 3 fields
          cmyk(_, _, _, _) { }
          no_color { }
        }
    }
}
