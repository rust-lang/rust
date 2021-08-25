// run-pass
// pretty-expanded FIXME #23616

fn leaky<T>(_t: T) { }

pub fn main() {
    let x = Box::new(10);
    leaky::<Box<isize>>(x);
}
