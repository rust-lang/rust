// xfail-stage0

fn iter_vec[T](&vec[T] v, &block (&T) f) {
    for (T x in v) {
        f(x);
    }
}

fn main() {
    auto v = [1,2,3,4,5];
    auto sum = 0;
    iter_vec(v, block (&int i)
    {
        iter_vec(v, block (&int j)
        {
            log_err i*j;
            sum += i*j;
        });
    });
    log_err sum;
    assert(sum == 225);
}
