// Regression test for #155482. If we allow the compiler to advance past `rustc_resolve`, it causes
// an ICE.
trait TraitA < AsA = impl TraitB < { //~ ERROR: cannot find trait `TraitB` in this scope
#[derive(Hash)]
  enum A; //~ ERROR: expected `{}`, found `;`
  struct A<A>; //~ ERROR: the name `A` is defined multiple times
}
>> ; //~ ERROR: expected `{}`, found `;`
