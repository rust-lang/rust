

enum foo<T> { arm(T), }

fn altfoo<T>(f: foo<T>) {
    let mut hit = false;
    match f { arm::<T>(x) => { debug!("in arm"); hit = true; } }
    assert (hit);
}

fn main() { altfoo::<int>(arm::<int>(10)); }
