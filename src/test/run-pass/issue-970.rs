enum maybe_ordered_pair {
    yes({low: int, high: int} : less_than(*.low, *.high));
    no;
}
pure fn less_than(x: int, y: int) -> bool { ret x < y; }
fn main() { }
