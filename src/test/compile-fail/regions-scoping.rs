// xfail-test

fn with<T>(t: T, f: fn(T)) { f(t) }

fn nested(x: &x.int) {  // (1)
    with(
        fn&(x: &x.int, // Refers to the region `x` at (1)
            y: &y.int, // A fresh region `y` (2)
            z: fn(x: &x.int, // Refers to `x` at (1)
                  y: &y.int, // Refers to `y` at (2)
                  z: &z.int) -> &z.int) // A fresh region `z` (3)
            -> &x.int {

            if false { ret z(x, x, x); } //! ERROR mismatched types: expected `&y.int` but found `&x.int`
            if false { ret z(x, x, y); } //! ERROR mismatched types: expected `&y.int` but found `&x.int`
                                        //!^ ERROR mismatched types: expected `&x.int` but found `&y.int`
            if false { ret z(x, y, x); }
            if false { ret z(x, y, y); } //! ERROR mismatched types: expected `&x.int` but found `&y.int`
            if false { ret z(y, x, x); } //! ERROR mismatched types: expected `&x.int` but found `&y.int`
                                        //!^ ERROR mismatched types: expected `&y.int` but found `&x.int`
            if false { ret z(y, x, y); } //! ERROR mismatched types: expected `&x.int` but found `&y.int`
                                        //!^ ERROR mismatched types: expected `&y.int` but found `&x.int`
                                       //!^^ ERROR mismatched types: expected `&x.int` but found `&y.int`
            if false { ret z(y, y, x); } //! ERROR mismatched types: expected `&x.int` but found `&y.int`
            if false { ret z(y, y, y); } //! ERROR mismatched types: expected `&x.int` but found `&y.int`
                                        //!^ ERROR mismatched types: expected `&x.int` but found `&y.int`
            fail;
        }
    ) {|f|

        let a: &x.int = f(x, x) { |_x, _y, z| z };
        let b: &x.int = f(x, a) { |_x, _y, z| z };
        let c: &x.int = f(a, a) { |_x, _y, z| z };

        let d: &x.int = f(x, x) { |_x, _y, z| z };
        let e: &x.int = f(x, &a) { |_x, _y, z| z };
        let f: &x.int = f(&a, &a) { |_x, _y, z| z };
    }
}

fn main() {}