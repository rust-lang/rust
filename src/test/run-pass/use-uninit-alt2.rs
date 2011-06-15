

fn foo[T](&myoption[T] o) -> int {
    let int x;
    alt (o) { case (none[T]) { fail; } case (some[T](?t)) { x = 5; } }
    ret x;
}

tag myoption[T] { none; some(T); }

fn main() { log 5; }