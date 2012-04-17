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
    ) {|foo|

        let a: &x.int = foo(x, x) { |_x, _y, z| z };
        let b: &x.int = foo(x, a) { |_x, _y, z| z };
        let c: &x.int = foo(a, a) { |_x, _y, z| z };

        let z = 3;
        let d: &x.int = foo(x, x) { |_x, _y, z| z };
        let e: &x.int = foo(x, &z) { |_x, _y, z| z };
        let f: &x.int = foo(&z, &z) { |_x, _y, z| z }; //! ERROR mismatched types: expected `&x.int` but found

        foo(x, &z) { |x, _y, _z| x }; //! ERROR mismatched types: expected `&z.int` but found `&x.int`
        foo(x, &z) { |_x, y, _z| y }; //! ERROR mismatched types: expected `&z.int` but found `&<block at
    }
}

fn main() {}