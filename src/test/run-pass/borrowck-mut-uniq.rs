type ints = {sum: ~int, values: ~[int]};

fn add_int(x: &mut ints, v: int) {
    *x.sum += v;
    let mut values = ~[];
    x.values <-> values;
    vec::push(values, v);
    x.values <- values;
}

fn iter_ints(x: &ints, f: fn(x: &int) -> bool) {
    let l = x.values.len();
    uint::range(0, l, |i| f(&x.values[i]))
}

fn main() {
    let mut ints = ~{sum: ~0, values: ~[]};
    add_int(ints, 22);
    add_int(ints, 44);

    for iter_ints(ints) |i| {
        error!{"int = %d", *i};
    }

    error!{"ints=%?", ints};
}
