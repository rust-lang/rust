// error-pattern:assigning to upvar
fn force(f: &block() -> int) -> int { ret f(); }
fn main() {
    let x = 5;
    let f = lambda () -> int { let y = 6; x <- y; ret 7 };
    assert (force(f) == 7);
    log x;
}
