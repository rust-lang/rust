fn something(f: pure fn()) { f(); }

fn main() {
    let mut x = ~[];
    something(|| vec::push(x, 0) ); //~ ERROR access to impure function prohibited in pure context
}
