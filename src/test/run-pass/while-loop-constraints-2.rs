// xfail-stage0
fn main() {

  let int y = 42;
  let int z = 42;
  let int x;
  while (z < 50) {
    z += 1; 
    while (false) {
      x <- y;
      y = z;
    }
    log y;
  }
  assert (y == 42 && z == 50);
}