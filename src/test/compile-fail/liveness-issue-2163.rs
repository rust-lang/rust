fn main(_s: [str]) {
    let a: [int] = [];
    vec::each(a) { |_x| //! ERROR not all control paths return a value
    }
}
