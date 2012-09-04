#[abi = "cdecl"];
#[link_name = "rustrt"];
#[link(name = "anonexternmod",
       vers = "0.1")];

#[crate_type = "lib"];
extern {
  fn last_os_error() -> ~str;
}
