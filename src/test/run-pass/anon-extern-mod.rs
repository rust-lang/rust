#[abi = "cdecl"]
#[link_name = "rustrt"]
extern {
  fn last_os_error() -> ~str;
}

fn main() {
  last_os_error();
}
