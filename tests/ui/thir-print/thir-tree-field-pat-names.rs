//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree
//@ normalize-stdout: "\[[a-z0-9]{4}\]" -> ""
//@ normalize-stdout: "DefId\(\d+:\d+" -> "DefId(N:M"


// Previously, `Leaf` and `Variant` patterns, when appearing within an irrefutable
// pattern only, did not show the field names, leaving the structure ambiguous.


struct Tri { x: u32, y: u32, z: u32 }

const K: u32 = 12;

enum Var {
  SLike { l: u32, r: u32 },
  Other (),
}


fn tuple_leaf (t: (u32, u32, u32)) {
  let t2 = (1, t, 2);

  let (a, (.., c), _) = t2;

  // refutable (different print path)
  if let (a, (K, ..), _) = t2 { }
}

fn struct_leaf (s: Tri) {
  let s2 = s;

  let Tri { x: a, y: b, z: c } = s2;
  let Tri { z: d, y: e, .. } = s2;

  // refutable (different print path)
  if let Tri { y: K, x, .. } = s2 {}
}

fn variant (e: Var) {
  // same print path as let-sans-else
  let Var::SLike { r: y, .. } = e else { unsafe { panic!() }};

  // different print path
  if let Var::SLike { r: y, .. } = e {}
}

fn main() {}
