


// -*- rust -*-
fn len(vec[mutable? int] v) -> uint {
    auto i = 0u;
    for (int x in v) { i += 1u; }
    ret i;
}

fn main() {
    auto v0 = [1, 2, 3, 4, 5];
    log len(v0);
    auto v1 = [mutable 1, 2, 3, 4, 5];
    log len(v1);
}