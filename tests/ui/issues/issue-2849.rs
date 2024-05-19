enum Foo { Alpha, Beta(isize) }

fn main() {
    match Foo::Alpha {
      Foo::Alpha | Foo::Beta(i) => {}
      //~^ ERROR variable `i` is not bound in all patterns
    }
}
