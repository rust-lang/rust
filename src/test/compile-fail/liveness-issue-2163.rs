fn main(s: [str]) {
    let a: [int] = [];
    vec::each(a) { |x| //! ERROR not all control paths return a value
    }
}
