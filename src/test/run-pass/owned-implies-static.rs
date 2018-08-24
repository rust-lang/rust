// pretty-expanded FIXME #23616

fn f<T: 'static>(_x: T) {}

pub fn main() {
    f(Box::new(5));
}
