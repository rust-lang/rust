trait Marker<const N: usize> {}
struct Example<const N: usize>;
impl<const N: usize> Marker<N> for Example<N> {}

fn make_marker() -> impl Marker<gimme_a_const!(marker)> {
  //~^ ERROR: type provided when a constant was expected
  //~| ERROR: type provided when a constant was expected
  Example::<gimme_a_const!(marker)>
  //~^ ERROR: type provided when a constant was expected
}

fn main() {
  let _ok = Example::<{
    #[macro_export]
    macro_rules! gimme_a_const {
      ($rusty: ident) => {{ let $rusty = 3; *&$rusty }}
      //~^ ERROR expected type
      //~| ERROR expected type
    }
    gimme_a_const!(run)
  }>;
  let _ok = Example::<{gimme_a_const!(marker)}>;
}
