// xfail-stage0

fn force(&block() -> int f) -> int { ret f(); }
fn main() {
    auto f = fn() -> int { ret 7 };
    assert(force(f) == 7);
    auto g = bind force(f);
    assert(g() == 7);
}
