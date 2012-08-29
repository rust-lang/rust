#[abi = "cdecl"]
#[link_name = "rustrt"]
extern mod {
  fn last_os_error() -> ~str;
}

fn main() {
  last_os_error();
}