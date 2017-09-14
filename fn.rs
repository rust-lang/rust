
fn foo(x:  fn(&u8, &u8), y: Vec<&u8>, z: &u8) {
// Debruijn   1    1            1        1
// Anon-Index 0    1            0        1
//            ------
//            debruijn indices are shifted by 1 in here
  y.push(z); // index will be zero or one
}
