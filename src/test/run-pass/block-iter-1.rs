// xfail-stage0

fn iter_vec[T](&vec[T] v, &block (&T) f) {
    for (T x in v) {
        f(x);
    }
}

fn main() {
    auto v = [1,2,3,4,5,6,7];
    auto odds = 0;
    iter_vec(v,
             block (&int i) {
                 log_err i;
                 if (i % 2 == 1) {
                     odds += 1;
                 }
                 log_err odds;
             });
    log_err odds;
    assert(odds == 4);
}
