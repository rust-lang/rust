// xfail-stage1
// xfail-stage2
// xfail-stage3

use std;
import std::sio;
import std::task;
import std::str;

fn connectTask(cx: sio::ctx, ip: str, portnum: int) {
  let client: sio::client;
  client = sio::connect_to(cx, ip, portnum);
  let data = sio::read(client);
  sio::close_client(client);
}

fn main() {
  let cx: sio::ctx = sio::new();
  let srv: sio::server = sio::create_server(cx, "0.0.0.0", 9090);
  let child: task = spawn connectTask(cx, "127.0.0.1", 9090);
  let client: sio::client = sio::accept_from(srv);
  sio::write_data(client, str::bytes("hello, world\n"));
  task::join(child);
  sio::close_client(client);
  sio::close_server(srv);
  sio::destroy(cx);
}


