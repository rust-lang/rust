// test that autoderef of a type like this does not
// cause compiler to loop.  Note that no instances
// of such a type could ever be constructed.
class t { //~ ERROR this type cannot be instantiated
  let x: x;
  let to_str: ();
  new(x: x) { self.x = x; self.to_str = (); }
}
enum x = @t; //~ ERROR this type cannot be instantiated

fn main() {
}
