trait T {
    fn f(self);
  }
  impl T for () {
    fn f(self) {
        let self = (); //~ ERROR expected unit struct, unit variant or constant, found local variable `self`
    }
}

fn main() {}
