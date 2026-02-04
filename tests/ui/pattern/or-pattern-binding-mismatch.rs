//! regression test for <https://github.com/rust-lang/rust/issues/2849>
enum Foo { Alpha, Beta(isize) }

fn main() {
    match Foo::Alpha {
      Foo::Alpha | Foo::Beta(i) => {}
      //~^ ERROR variable `i` is not bound in all patterns
    }
}
