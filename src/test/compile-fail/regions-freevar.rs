fn wants_static_fn(_x: &static/fn()) {}

fn main() {
    let i = 3;
    do wants_static_fn {
        debug!("i=%d", i);
          //~^ ERROR captured variable does not outlive the enclosing closure
    }
}
