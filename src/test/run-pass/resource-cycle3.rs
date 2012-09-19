// same as resource-cycle2, but be sure to give r multiple fields... 

// Don't leak the unique pointers

type u = {
    a: int,
    b: int,
    c: *int
};

struct r {
  v: u,
  w: int,
  x: *int,
  drop unsafe {
    let _v2: ~int = cast::reinterpret_cast(&self.v.c);
    // let _v3: ~int = unsafe::reinterpret_cast(self.x);
  }
}

fn r(v: u, w: int, _x: *int) -> r unsafe {
    r {
        v: v,
        w: w,
        x: cast::reinterpret_cast(&0)
    }
}

enum t = {
    mut next: Option<@t>,
    r: r
};

fn main() unsafe {
    let i1 = ~0xA;
    let i1p = cast::reinterpret_cast(&i1);
    cast::forget(move i1);
    let i2 = ~0xA;
    let i2p = cast::reinterpret_cast(&i2);
    cast::forget(move i2);

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
