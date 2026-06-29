//! Regression test for <https://github.com/rust-lang/rust/issues/24363>.

fn main() {
    1.create_a_type_error[ //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields
        ()+() //~ ERROR cannot add
              //   ^ ensure that we typeck the inner expression ^
    ];
}
