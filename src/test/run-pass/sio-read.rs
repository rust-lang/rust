// xfail-test
use std;
import std::sio;
import std::task;
import std::str;
import std::net;

fn connectTask(cx: sio::ctx, ip: net::ip_addr, portnum: int) {
  let client: sio::client;
  client = sio::connect_to(cx, ip, portnum);
  let data = sio::read(client);
  sio::close_client(client);
}

fn main() {
  let cx: sio::ctx = sio::new();
  let srv: sio::server = sio::create_server(
          cx, net::parse_addr("0.0.0.0"), 9090);
  let child = task::_spawn(bind connectTask(cx,
                                            net::parse_addr("127.0.0.1"),
                                            9090));
  let client: sio::client = sio::accept_from(srv);
  sio::write_data(client, str::bytes("hello, world\n"));
  task::join_id(child);
  sio::close_client(client);
  sio::close_server(srv);
  sio::destroy(cx);
}


