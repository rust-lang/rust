//@ check-pass

// Ensure that `dyn $($AutoTrait)+ ObjSafe` is well-formed.

// Some arbitrary dyn-compatible trait:
trait Obj {}

type _0 = dyn Unpin;
type _1 = dyn Send + Obj;
type _2 = dyn Send + Unpin + Obj;
type _3 = dyn Send + Unpin + Sync + Obj;

fn main() {}
