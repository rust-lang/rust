use sys::size_of;
extern mod std;

struct Cat {
    x: int
}

struct Kitty {
    x: int,
}

impl Kitty : Drop {
    fn finalize(&self) {}
}

#[cfg(target_arch = "x86_64")]
fn main() {
    assert (size_of::<Cat>() == 8 as uint);
    assert (size_of::<Kitty>() == 16 as uint);
}

#[cfg(target_arch = "x86")]
fn main() {
    assert (size_of::<Cat>() == 4 as uint);
    assert (size_of::<Kitty>() == 8 as uint);
}
