// error-pattern:fail

fn failfn() {
    fail;
}

class r {
  let v: *int;
  new(v: *int) { self.v = v; }
  drop unsafe {
    let _v2: ~int = unsafe::reinterpret_cast(self.v);
  }
}

fn main() unsafe {
    let i1 = ~0;
    let i1p = unsafe::reinterpret_cast(i1);
    unsafe::forget(i1);
    let x = @r(i1p);
    failfn();
    log(error, x);
}