tag t1 { a(int); b(uint); }
type t2 = {x: t1, y: int};
tag t3 { c(t2, uint); }

fn m(in: &t3) -> int {
    alt in {
      c({x: a(m), _}, _) { ret m; }
      c({x: b(m), y: y}, z) { ret (m + z as int) + y; }
    }
}

fn main() {
    assert (m(c({x: a(10), y: 5}, 4u)) == 10);
    assert (m(c({x: b(10u), y: 5}, 4u)) == 19);
}
