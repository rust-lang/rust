// xfail-stage0

fn add1(int i) -> int { ret i+1; }
fn main() {
    auto f = @add1;
    auto g = @f;
    auto h = @@@add1;
    assert(f(5) == 6);
    assert(g(8) == 9);
    assert(h(0x1badd00d) == 0x1badd00e);
}
