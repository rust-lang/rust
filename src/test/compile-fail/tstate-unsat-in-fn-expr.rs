fn foo(v: [int]) : vec::is_empty(v) { #debug("%d", v[0]); }

fn main() {
    let f = fn@() {
        let v = ~[1];
        foo(v); //! ERROR unsatisfied precondition constraint
    };
    log(error, f());
}
