#[link(name = "req")];
#[crate_type = "lib"];

use std;

use dvec::*;
use dvec::DVec;
use std::map::hashmap;

type header_map = hashmap<~str, @DVec<@~str>>;

// the unused ty param is necessary so this gets monomorphized
fn request<T: Copy>(req: header_map) {
  let _x = *(*req.get(~"METHOD"))[0u];
}
