// run-pass

macro_rules! sty {
    ($t:ty) => (stringify!($t))
}

macro_rules! spath {
    ($t:path) => (stringify!($t))
}

fn main() {
    assert_eq!(sty!(isize), "isize");
    assert_eq!(spath!(std::option), "std::option");
}
