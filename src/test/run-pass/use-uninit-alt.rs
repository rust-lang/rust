

fn foo[T](&myoption[T] o) -> int {
    let int x = 5;
    alt (o) { case (none[T]) { } case (some[T](?t)) { x += 1; } }
    ret x;
}

tag myoption[T] { none; some(T); }

fn main() { log 5; }