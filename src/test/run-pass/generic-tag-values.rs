


// -*- rust -*-
tag noption[T] { some(T); }

fn main() {
    let nop: noption<int> = some[int](5);
    alt nop { some[int](n) { log n; assert (n == 5); } }
    let nop2: noption<{x: int, y: int}> = some({x: 17, y: 42});
    alt nop2 {
      some(t) { log t.x; log t.y; assert (t.x == 17); assert (t.y == 42); }
    }
}
