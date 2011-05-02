fn main() {
    auto s = "a";
    s += "b";
    check (s.(0) == ('a' as u8));
    check (s.(1) == ('b' as u8));

    s += "c";
    s += "d";
    check (s.(0) == ('a' as u8));
    check (s.(1) == ('b' as u8));
    check (s.(2) == ('c' as u8));
    check (s.(3) == ('d' as u8));
}

