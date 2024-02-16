//@ check-pass

trait T {}

fn wrap(x: impl T) -> impl T {
    //~^ WARN function cannot return without recursing
    wrap(wrap(x))
}

fn main() {}
