// Don't leak the unique pointers

struct r {
  let v: *int;
  new(v: *int) unsafe {
    self.v = v;
    debug!{"r's ctor: v = %x, self = %x, self.v = %x",
           unsafe::reinterpret_cast::<*int, uint>(v),
           unsafe::reinterpret_cast::<*r, uint>(ptr::addr_of(self)),
           unsafe::reinterpret_cast::<**int, uint>(ptr::addr_of(self.v))};
     }
  drop unsafe {
    debug!{"r's dtor: self = %x, self.v = %x, self.v's value = %x",
           unsafe::reinterpret_cast::<*r, uint>(ptr::addr_of(self)),
           unsafe::reinterpret_cast::<**int, uint>(ptr::addr_of(self.v)),
           unsafe::reinterpret_cast::<*int, uint>(self.v)};
    let v2: ~int = unsafe::reinterpret_cast(self.v); }
}

enum t = {
    mut next: option<@t>,
    r: r
};

fn main() unsafe {
    let i1 = ~0;
    let i1p = unsafe::reinterpret_cast(i1);
    unsafe::forget(i1);
    let i2 = ~0;
    let i2p = unsafe::reinterpret_cast(i2);
    unsafe::forget(i2);

    let x1 = @t({
        mut next: none,
          r: {
          let rs = r(i1p);
          debug!{"r = %x",
                 unsafe::reinterpret_cast::<*r, uint>(ptr::addr_of(rs))};
          rs }
    });
    
    debug!{"x1 = %x, x1.r = %x",
        unsafe::reinterpret_cast::<@t, uint>(x1),
        unsafe::reinterpret_cast::<*r, uint>(ptr::addr_of(x1.r))};

    let x2 = @t({
        mut next: none,
          r: {
          let rs = r(i2p);
          debug!{"r2 = %x",
                 unsafe::reinterpret_cast::<*r, uint>(ptr::addr_of(rs))};
          rs
            }
    });
    
    debug!{"x2 = %x, x2.r = %x",
           unsafe::reinterpret_cast::<@t, uint>(x2),
           unsafe::reinterpret_cast::<*r, uint>(ptr::addr_of(x2.r))};

    x1.next = some(x2);
    x2.next = some(x1);
}
