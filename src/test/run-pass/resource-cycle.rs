// Don't leak the unique pointers

struct r {
  v: *int,
}

impl r : Drop {
    fn finalize() {
        unsafe {
            debug!("r's dtor: self = %x, self.v = %x, self.v's value = %x",
              cast::reinterpret_cast::<*r, uint>(&ptr::addr_of(&self)),
              cast::reinterpret_cast::<**int, uint>(&ptr::addr_of(&(self.v))),
              cast::reinterpret_cast::<*int, uint>(&self.v));
            let v2: ~int = cast::reinterpret_cast(&self.v);
        }
    }
}

fn r(v: *int) -> r unsafe {
    r {
        v: v
    }
}

enum t = {
    mut next: Option<@t>,
    r: r
};

fn main() unsafe {
    let i1 = ~0;
    let i1p = cast::reinterpret_cast(&i1);
    cast::forget(move i1);
    let i2 = ~0;
    let i2p = cast::reinterpret_cast(&i2);
    cast::forget(move i2);

    let x1 = @t({
        mut next: None,
          r: {
          let rs = r(i1p);
          debug!("r = %x",
                 cast::reinterpret_cast::<*r, uint>(&ptr::addr_of(&rs)));
          move rs }
    });
    
    debug!("x1 = %x, x1.r = %x",
        cast::reinterpret_cast::<@t, uint>(&x1),
        cast::reinterpret_cast::<*r, uint>(&ptr::addr_of(&(x1.r))));

    let x2 = @t({
        mut next: None,
          r: {
          let rs = r(i2p);
          debug!("r2 = %x",
                 cast::reinterpret_cast::<*r, uint>(&ptr::addr_of(&rs)));
          move rs
            }
    });
    
    debug!("x2 = %x, x2.r = %x",
           cast::reinterpret_cast::<@t, uint>(&x2),
           cast::reinterpret_cast::<*r, uint>(&ptr::addr_of(&(x2.r))));

    x1.next = Some(x2);
    x2.next = Some(x1);
}
