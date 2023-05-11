// check that the local data keys are private by default.

mod bar {
    thread_local!(static baz: f64 = 0.0);
}

fn main() {
    bar::baz.with(|_| ());
    //~^ ERROR `baz` is private
}
