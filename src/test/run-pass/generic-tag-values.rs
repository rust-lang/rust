


// -*- rust -*-
tag noption<T> { some(T); }

fn main() {
    let nop: noption<int> = some::<int>(5);
    alt nop { some::<int>(n) { log_full(core::debug, n); assert (n == 5); } }
    let nop2: noption<{x: int, y: int}> = some({x: 17, y: 42});
    alt nop2 {
      some(t) {
        log_full(core::debug, t.x);
        log_full(core::debug, t.y);
        assert (t.x == 17);
        assert (t.y == 42);
      }
    }
}
