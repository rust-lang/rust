//aux-build:extern-crosscrate-source.rs

use externcallback(vers = "0.1");

fn fact(n: uint) -> uint {
    #debug("n = %?", n);
    externcallback::rustrt::rust_dbg_call(externcallback::cb, n)
}

fn main() {
    let result = fact(10u);
    #debug("result = %?", result);
    assert result == 3628800u;
}
