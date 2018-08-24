enum f { g(isize, isize) }

enum h { i(j, k) }

enum j { l(isize, isize) }
enum k { m(isize, isize) }

fn main()
{

    let _z = match f::g(1, 2) {
      f::g(x, x) => { println!("{}", x + x); }
      //~^ ERROR identifier `x` is bound more than once in the same pattern
    };

    let _z = match h::i(j::l(1, 2), k::m(3, 4)) {
      h::i(j::l(x, _), k::m(_, x))
      //~^ ERROR identifier `x` is bound more than once in the same pattern
        => { println!("{}", x + x); }
    };

    let _z = match (1, 2) {
        (x, x) => { x } //~ ERROR identifier `x` is bound more than once in the same pattern
    };

}
