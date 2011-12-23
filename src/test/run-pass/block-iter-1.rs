fn iter_vec<T>(v: [T], f: block(T)) { for x: T in v { f(x); } }

fn main() {
    let v = [1, 2, 3, 4, 5, 6, 7];
    let odds = 0;
    iter_vec(v, {|i|
        log(error, i);
        if i % 2 == 1 {
            odds += 1;
        }
        log(error, odds);
    });
    log(error, odds);
    assert (odds == 4);
}
