// A complex case with mixed suggestions from #65853

enum E { X, Y }
enum F { X2, Y2 }
struct G {}
struct H {}
struct X {}
struct Y {}
struct Z {}

fn complex(_i: u32, _s: &str, _e: E, _f: F, _g: G, _x: X, _y: Y, _z: Z ) {}

fn main() {
  complex(1.0, H {}, &"", G{}, F::X2, Z {}, X {}, Y {});
  //~^ ERROR arguments to this function are incorrect
}
