// xfail-test
trait Positioned {
  fn SetX(int);
}

trait Movable: Positioned {
  fn translate(dx: int) {
    self.SetX(self.X() + dx);
  }
}

fn main() {}
