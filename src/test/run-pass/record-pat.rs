tag t1 { a(int); b(uint); }
type t2 = rec(t1 x, int y);
tag t3 { c(t2, uint); }

fn m(&t3 in) -> int {
    alt in {
        c({x: a(?m), _}, _) { ret m; }
        c({x: b(?m), y}, ?z) { ret (m + z) as int + y; }
    }
}

fn main() {
    assert m(c(rec(x=a(10), y=5), 4u)) == 10;
    assert m(c(rec(x=b(10u), y=5), 4u)) == 19;
}
