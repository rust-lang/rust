// pretty-expanded FIXME #23616

static mut n_mut: usize = 0;

static n: &'static usize = unsafe{ &n_mut };

fn main() {}
