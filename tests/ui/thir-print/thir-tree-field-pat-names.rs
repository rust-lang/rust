//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree
//@ normalize-stdout: "\[[a-z0-9]{4}\]" -> ""
//@ normalize-stdout: "DefId\(\d+:\d+" -> "DefId(N:M"


// Previously, `Leaf` and `Variant` patterns, when appearing within an irrefutable
// pattern only, did not show the field names, leaving the structure ambiguous.


struct Tri { x: u32, y: u32, z: u32 }

fn tuple_leaf (t: (u32, u32, u32)) {
  let (a, ..) = t;

  // refutable (different print path)
  if let (a, 12, ..) = t { }
}

fn struct_leaf (s: Tri) {
  let Tri { z: d, y: e, .. } = s;

  // refutable (different print path)
  if let Tri { y: 12, x, .. } = s {}
}

fn main() {}
