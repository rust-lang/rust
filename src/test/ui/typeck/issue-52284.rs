struct Alef();
struct Bet();
fn gimel() -> Alef {
  if true {
    Alef() //~ERROR
  }
  if true {
    Bet() //~ERROR
  }
  Alef()
}
fn main() {
  gimel();
}
