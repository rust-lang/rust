enum f { g(int, int) }

enum h { i(j, k) }

enum j { l(int, int) }
enum k { m(int, int) }

fn main()
{

    let _z = match g(1, 2) {
      g(x, x) => { log(debug, x + x); }
      //~^ ERROR Identifier x is bound more than once in the same pattern
    };

    let _z = match i(l(1, 2), m(3, 4)) {
      i(l(x, _), m(_, x))  //~ ERROR Identifier x is bound more than once in the same pattern
        => { log(error, x + x); }
    };

    let _z = match (1, 2) {
        (x, x) => { x } //~ ERROR Identifier x is bound more than once in the same pattern
    };

}
