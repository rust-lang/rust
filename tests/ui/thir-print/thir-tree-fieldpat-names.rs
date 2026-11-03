//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree


struct Tri { x: u32, y: u32, z: u32 }

const K: u32 = 12;

enum Var {
  SLike { l: u32, r: u32 },
  Other (),
}


fn tuple_leaf (t: (u32, u32, u32)) {
  let t2 = (1, t, 2);

  let (a, (_, b, _), c) = t2;

  let (a, (b, ..), _) = t2;
  let (a, (.., c), _) = t2;

  let (a, .., e) = t;

  // refutable (different print path)
  if let (a, (K, ..), _) = t2 { }
}

fn struct_leaf (s: Tri) {
  let s2 = s;

  let Tri { x, y, z } = s2;
  let Tri { z, y, x } = s2;

  let Tri { x: a, y: b, z: c } = s2;
  let Tri { z: d, y: e, x: f } = s2;
  let Tri { y, .. } = s2;
  let Tri { z: c, .. } = s2;

  // refutable (different print path)
  if let Tri { y: K, x, .. } = s2 {}
}

fn variant (e: Var) {
  // same print path as let-sans-else
  let Var::SLike { l, r: _ } = e else { unsafe { mypanic() }};
  let Var::SLike { r: y, .. } = e else { unsafe { mypanic() }};

  // different print path
  if let Var::SLike { r: y, .. } = e {}
}

fn main() {}

unsafe extern "C" { // no std in stdout
  fn mypanic() -> !;
}
