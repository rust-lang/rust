use std::mem;

struct A { x: i32, y: f64 }

#[cfg(not(FALSE))]
unsafe fn access(n:*mut A) -> (i32, f64) {
    let x : i32 = n.x; //~ ERROR no field `x` on type `*mut A`
    let y : f64 = n.y; //~ ERROR no field `y` on type `*mut A`
    (x, y)
}

#[cfg(false)]
unsafe fn access(n:*mut A) -> (i32, f64) {
    let x : i32 = (*n).x;
    let y : f64 = (*n).y;
    (x, y)
}

fn main() {
    let a :  A = A { x: 3, y: 3.14 };
    let p : &A = &a;
    let (x,y) = unsafe {
        let n : *mut A = mem::transmute(p);
        access(n)
    };
    println!("x: {}, y: {}", x, y);
}
