use io::println;

fn main() {
  trait Text {
    fn to_str(&self) -> ~str;
  }

  fn to_string(t: Text) {
    println(t.to_str());
  }

}
