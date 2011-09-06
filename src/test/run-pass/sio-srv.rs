// xfail-test
use std;
import std::sio;
import std::net;

fn main() {
  let cx: sio::ctx = sio::new();
  let srv: sio::server = sio::create_server(cx,
                                            net::parse_addr("127.0.0.1"),
                                            9090);
  sio::close_server(srv);
  sio::destroy(cx);
}
