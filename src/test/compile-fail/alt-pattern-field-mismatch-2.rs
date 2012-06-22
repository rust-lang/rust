fn main() {
    enum color {
        rgb(uint, uint, uint),
        cmyk(uint, uint, uint, uint),
        no_color,
    }

    fn foo(c: color) {
        alt c {
          rgb(_, _, _) { }
          cmyk(_, _, _, _) { }
          no_color(_) { }
          //!^ ERROR this pattern has 1 field, but the corresponding variant has no fields
        }
    }
}
