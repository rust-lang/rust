// pretty-compare-only
// pretty-mode:expanded
// pp-exact:cast-lt.pp

macro_rules! negative {
      ($e:expr) => { $e < 0 }
}

fn main() {
      negative!(1 as i32);
}
