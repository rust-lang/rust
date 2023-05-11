enum F { G(isize, isize) }

enum H { I(J, K) }

enum J { L(isize, isize) }
enum K { M(isize, isize) }

fn main()
{

    let _z = match F::G(1, 2) {
      F::G(x, x) => { println!("{}", x + x); }
      //~^ ERROR identifier `x` is bound more than once in the same pattern
    };

    let _z = match H::I(J::L(1, 2), K::M(3, 4)) {
      H::I(J::L(x, _), K::M(_, x))
      //~^ ERROR identifier `x` is bound more than once in the same pattern
        => { println!("{}", x + x); }
    };

    let _z = match (1, 2) {
        (x, x) => { x } //~ ERROR identifier `x` is bound more than once in the same pattern
    };

}
