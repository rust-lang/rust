fn main(s: [str]) {
    let a: [int] = [];
    vec::each(a) { |x| //! ERROR in function `anon`, not all control paths
    }                  //! ERROR see function return type of `bool`
}
