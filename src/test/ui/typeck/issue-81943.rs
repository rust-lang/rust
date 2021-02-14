fn f<F: Fn(i32)>(f: F) { f(0); }
fn main() {
  f(|x| dbg!(x)); //~ERROR
  f(|x| match x { tmp => { tmp } }); //~ERROR
  macro_rules! d {
    ($e:expr) => { match $e { x => { x } } }
  }
  f(|x| d!(x)); //~ERROR
}
