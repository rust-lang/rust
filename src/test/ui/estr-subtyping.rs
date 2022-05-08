fn wants_uniq(x: String) { }
fn wants_slice(x: &str) { }

fn has_uniq(x: String) {
   wants_uniq(x);
   wants_slice(&*x);
}

fn has_slice(x: &str) {
   wants_uniq(x); //~ ERROR mismatched types
   wants_slice(x);
}

fn main() {
}
