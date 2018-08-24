// pretty-expanded FIXME #23616

static foo: [usize; 3] = [1, 2, 3];

static slice_1: &'static [usize] = &foo;
static slice_2: &'static [usize] = &foo;

fn main() {}
