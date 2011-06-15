

tag foo[T] { arm(T); }

fn altfoo[T](&foo[T] f) {
    auto hit = false;
    alt (f) { case (arm[T](?x)) { log "in arm"; hit = true; } }
    assert (hit);
}

fn main() { altfoo[int](arm[int](10)); }