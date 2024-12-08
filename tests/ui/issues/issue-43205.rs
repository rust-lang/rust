//@ run-pass
fn main() {
   let _ = &&[()][0];
   println!("{:?}", &[(),()][1]);
}
