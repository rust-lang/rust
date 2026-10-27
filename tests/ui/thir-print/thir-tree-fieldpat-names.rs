//@ check-pass
//@ compile-flags: -Zunpretty=thir-tree

struct Sts (u32, u32, u32);
struct Tri { x: u32, y: u32, z: u32 }

const J: u32 = 11;
const K: u32 = 12;

enum Var {
  TLike (u32, u32),
  SLike { l: u32, r: u32 },
}


fn tuple_leaf (t: (u32, u32, u32)) {
  let t2 = (1, t, 2);

  let (a, (_, b, _), c) = t2;
  let (a, tn @ (_, b, _), c) = t2;

  let (a, (b, ..), _) = t2;
  let (a, (.., c), _) = t2;

  let t3 = (1, 2, 3, 4, 5);
  let (a, .., e) = t3;

  // refutable (different print path)
  if let (a, (_, J, _), c) = t2 { }
  if let (a, (_, b, _), K) = t2 { }
  if let (a, (K, ..), _) = t2 { }
}

fn struct_leaf (s: Tri) {
  let s2 = s;

  let Tri { x, y, z } = s2;
  let Tri { z, y, x } = s2;
  let Tri { x, y: _, z } = s2;
  let Tri { z, y: _, x } = s2;

  let Tri { x: a, y: b, z: c } = s2;
  let Tri { z: d, y: e, x: f } = s2;

  let Tri { x, .. } = s2;
  let Tri { y, .. } = s2;
  let Tri { z: c, .. } = s2;

  // refutable (different print path)
  if let Tri { x: J, y, .. } = s2 {}
  if let Tri { y: K, x, .. } = s2 {}
  if let Tri { z: d, y: e, x: K } = s2 {}
}

fn variant (e: Var) {
  let Var::TLike (x, _) = e else { unsafe { mypanic() }};
  let Var::TLike (_, y) = e else { unsafe { mypanic() }};
  let Var::TLike (_, _) = e else { unsafe { mypanic() }};

  let Var::TLike (.., y) = e else { unsafe { mypanic() }};
  let Var::TLike (x, ..) = e else { unsafe { mypanic() }};

  let Var::SLike { l, r: _ } = e else { unsafe { mypanic() }};
  let Var::SLike { l: _, r } = e else { unsafe { mypanic() }};
  let Var::SLike { l: _, r: _ } = e else { unsafe { mypanic() }};

  let Var::SLike { r: y, .. } = e else { unsafe { mypanic() }};
  let Var::SLike { l: x, .. } = e else { unsafe { mypanic() }};

  // refutable (different print path)
  if let Var::TLike (.., y) = e {}
  if let Var::SLike { r: y, .. } = e {}
  if let Var::SLike { l: x, .. } = e {}
}

fn main() {}

unsafe extern "C" { // no std in stdout
  fn mypanic() -> !;
}
