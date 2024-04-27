fn main() {
  let _ = fix(|_: &dyn Fn()| {});
}

fn fix<F: Fn(G), G: Fn()>(f: F) -> impl Fn() {
  move || f(fix(&f))
  //~^ ERROR concrete type differs from previous defining opaque type use
  //~| ERROR expected generic type parameter, found `&F`
}
