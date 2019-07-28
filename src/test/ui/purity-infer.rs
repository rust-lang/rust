// run-pass

fn something<F>(f: F) where F: FnOnce() { f(); }
pub fn main() {
    something(|| println!("hi!") );
}
