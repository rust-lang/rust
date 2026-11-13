// Check that borrowck ensures that `static` items have the expected type.

static FOO: &'static (dyn Fn(&'static u8) + Send + Sync) = &drop;

fn main() {
    let n = 42;
    FOO(&n);
    //~^ ERROR does not live long enough
}
