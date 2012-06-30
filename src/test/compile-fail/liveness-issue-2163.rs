// xfail-test After the closure syntax change this started failing with the wrong error message
fn main(_s: ~[str]) {
    let a: ~[int] = ~[];
    do vec::each(a) |_x| { //! ERROR not all control paths return a value
    }
}
