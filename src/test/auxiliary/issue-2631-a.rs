#[link(name = "req")];
#[crate_type = "lib"];

use std;

import dvec::*;
import dvec::dvec;
import std::map::hashmap;

type header_map = hashmap<~str, @dvec<@~str>>;

// the unused ty param is necessary so this gets monomorphized
fn request<T: copy>(req: header_map) {
  let _x = *(*req.get(~"METHOD"))[0u];
}
