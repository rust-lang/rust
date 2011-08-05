use std;
import std::sio;

fn main() {
  let cx: sio::ctx = sio::new();
  sio::destroy(cx);
}
