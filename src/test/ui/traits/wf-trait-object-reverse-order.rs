// run-pass

// Ensure that `dyn $($AutoTrait) + ObjSafe` is well-formed.

use std::marker::Unpin;

// Some arbitray object-safe trait:
trait Obj {}

type _0 = Unpin;
type _1 = Send + Obj;
type _2 = Send + Unpin + Obj;
type _3 = Send + Unpin + Sync + Obj;

fn main() {}
