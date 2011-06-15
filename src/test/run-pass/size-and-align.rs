


// -*- rust -*-
tag clam[T] { a(T, int); b; }

fn uhoh[T](vec[clam[T]] v) {
    alt (v.(1)) {
        case (a[T](?t, ?u)) { log "incorrect"; log u; fail; }
        case (b[T]) { log "correct"; }
    }
}

fn main() {
    let vec[clam[int]] v = [b[int], b[int], a[int](42, 17)];
    uhoh[int](v);
}