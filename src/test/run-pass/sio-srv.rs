use std;
import std::sio;

fn main() {
  let cx: sio::ctx = sio::new();
  let srv: sio::server = sio::create_server(cx, "0.0.0.0", 9090);
  sio::close_server(srv);
  sio::destroy(cx);
}
