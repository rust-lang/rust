// xfail-stage0

fn main() {
    #macro([#mylambda(x, body),
            {
                fn f(x: int) -> int { ret body }
                f
            }]);

    assert (#mylambda(y, y * 2)(8) == 16);
}