// same as resource-cycle2, but be sure to give r multiple fields... 

// Don't leak the unique pointers

type u = {
    a: int,
    b: int,
    c: *int
};

struct r {
  let v: u;
  let w: int;
  let x: *int;
  new(v: u, w: int, _x: *int) unsafe { self.v = v; self.w = w; 
    self.x = unsafe::reinterpret_cast(0);
    /* self.x = x; */ }
  drop unsafe {
    let _v2: ~int = unsafe::reinterpret_cast(self.v.c);
    // let _v3: ~int = unsafe::reinterpret_cast(self.x);
  }
}

enum t = {
    mut next: Option<@t>,
    r: r
};

fn main() unsafe {
    let i1 = ~0xA;
    let i1p = unsafe::reinterpret_cast(i1);
    unsafe::forget(i1);
    let i2 = ~0xA;
    let i2p = unsafe::reinterpret_cast(i2);
    unsafe::forget(i2);

    let u1 = {a: 0xB, b: 0xC, c: i1p};
    let u2 = {a: 0xB, b: 0xC, c: i2p};

    let x1 = @t({
        mut next: None,
        r: r(u1, 42, i1p)
    });
    let x2 = @t({
        mut next: None,
        r: r(u2, 42, i2p)
    });
    x1.next = Some(x2);
    x2.next = Some(x1);
}
