struct Foo;
fn takes_ref(_: &Foo) {}

fn main() {
  let ref opt = Some(Foo);
  opt.map(|arg| takes_ref(arg));
  //~^ ERROR mismatched types [E0308]
  opt.and_then(|arg| Some(takes_ref(arg)));
  //~^ ERROR mismatched types [E0308]
  let ref opt: Result<_, ()> = Ok(Foo);
  opt.map(|arg| takes_ref(arg));
  //~^ ERROR mismatched types [E0308]
  opt.and_then(|arg| Ok(takes_ref(arg)));
  //~^ ERROR mismatched types [E0308]
}
