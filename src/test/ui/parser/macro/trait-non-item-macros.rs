macro_rules! bah {
    ($a:expr) => ($a)
    //~^ ERROR expected one of `async`
}

trait bar {
    bah!(2);
}

fn main() {}
