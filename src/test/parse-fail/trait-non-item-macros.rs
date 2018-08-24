macro_rules! bah {
    ($a:expr) => ($a)
    //~^ ERROR expected one of `async`, `const`, `extern`, `fn`, `type`, or `unsafe`, found `2`
}

trait bar {
    bah!(2);
}

fn main() {}
