pure fn call_first((x, _y): (&fn(), &fn())) {
    x();    //~ ERROR access to impure function prohibited in pure context
}

fn main() {}

