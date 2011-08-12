

tag foo<T> { arm(T); }

fn altfoo<T>(f: &foo<T>) {
    let hit = false;
    alt f { arm[T](x) { log "in arm"; hit = true; } }
    assert (hit);
}

fn main() { altfoo[int](arm[int](10)); }
