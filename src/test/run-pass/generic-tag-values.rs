


// -*- rust -*-
tag noption[T] { some(T); }

fn main() {
    let noption[int] nop = some[int](5);
    alt (nop) { case (some[int](?n)) { log n; assert (n == 5); } }
    let noption[rec(int x, int y)] nop2 = some(rec(x=17, y=42));
    alt (nop2) {
        case (some(?t)) {
            log t.x;
            log t.y;
            assert (t.x == 17);
            assert (t.y == 42);
        }
    }
}