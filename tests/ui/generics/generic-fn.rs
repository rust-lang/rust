// run-pass
#![allow(dead_code)]
#![allow(unused_assignments)]

fn id<T>(x: T) -> T { return x; }

#[derive(Copy, Clone)]
struct Triple {x: isize, y: isize, z: isize}

pub fn main() {
    let mut x = 62;
    let mut y = 63;
    let a = 'a';
    let mut b = 'b';
    let p: Triple = Triple {x: 65, y: 66, z: 67};
    let mut q: Triple = Triple {x: 68, y: 69, z: 70};
    y = id::<isize>(x);
    println!("{}", y);
    assert_eq!(x, y);
    b = id::<char>(a);
    println!("{}", b);
    assert_eq!(a, b);
    q = id::<Triple>(p);
    x = p.z;
    y = q.z;
    println!("{}", y);
    assert_eq!(x, y);
}
