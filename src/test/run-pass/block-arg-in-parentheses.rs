fn w_semi(v: [int]) -> int {
    vec::foldl(0, v) {|x,y| x+y};
    -10
}

fn wo_paren(v: [int]) -> int {
    // Perhaps surprising: this is parsed equivalently to w_semi()
    vec::foldl(0, v) {|x,y| x+y} - 10
}

fn w_paren(v: [int]) -> int {
    // Here the parentheses force interpretation as an expression:
    (vec::foldl(0, v) {|x,y| x+y}) - 10
}

fn main() {
    assert wo_paren([0, 1, 2, 3]) == -10;
    assert w_semi([0, 1, 2, 3]) == -10;
    assert w_paren([0, 1, 2, 3]) == -4;
}

