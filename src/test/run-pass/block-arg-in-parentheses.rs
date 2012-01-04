fn w_semi(v: [int]) -> int {
    // the semicolon causes compiler not to
    // complain about the ignored return value:
    vec::foldl(0, v) {|x,y| x+y};
    -10
}

fn w_paren1(v: [int]) -> int {
    (vec::foldl(0, v) {|x,y| x+y}) - 10
}

fn w_paren2(v: [int]) -> int {
    (vec::foldl(0, v) {|x,y| x+y} - 10)
}

fn w_ret(v: [int]) -> int {
    ret vec::foldl(0, v) {|x,y| x+y} - 10;
}

fn main() {
    assert w_semi([0, 1, 2, 3]) == -10;
    assert w_paren1([0, 1, 2, 3]) == -4;
    assert w_paren2([0, 1, 2, 3]) == -4;
    assert w_ret([0, 1, 2, 3]) == -4;
}

