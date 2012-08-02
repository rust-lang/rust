// Check that pure functions can modify local state.

pure fn sums_to(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = 0;
    while i < v.len() {
        sum0 += v[i];
        i += 1u;
    }
    return sum0 == sum;
}

pure fn sums_to_using_uniq(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = ~mut 0;
    while i < v.len() {
        *sum0 += v[i];
        i += 1u;
    }
    return *sum0 == sum;
}

pure fn sums_to_using_rec(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = {f: 0};
    while i < v.len() {
        sum0.f += v[i];
        i += 1u;
    }
    return sum0.f == sum;
}

pure fn sums_to_using_uniq_rec(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = {f: ~mut 0};
    while i < v.len() {
        *sum0.f += v[i];
        i += 1u;
    }
    return *sum0.f == sum;
}

fn main() {
}