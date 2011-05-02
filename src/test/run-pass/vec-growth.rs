fn main() {
    auto v = vec(1);
    v += vec(2);
    v += vec(3);
    v += vec(4);
    v += vec(5);
    check (v.(0) == 1);
    check (v.(1) == 2);
    check (v.(2) == 3);
    check (v.(3) == 4);
    check (v.(4) == 5);
}

