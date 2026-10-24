#[derive(Debug)]
struct X<T>(T);

impl<T: Clone> Clone for X<T> {
    fn clone(&self) -> X<T> {
        X(self.0.clone())
    }
}

#[derive(Debug)]
struct Y;

#[derive(Debug)]
struct Str {
   x: Option<i32>,
}

fn foo(s: &mut Option<i32>) {
    if s.is_none() {
        *s = Some(0);
    }
    println!("{:?}", s);
}

fn bar<T: std::fmt::Debug>(s: &mut X<T>) {
    println!("{:?}", s);
}
fn main() {
    let s = Str { x: None };
    let sr = &s;
    let mut sm = sr.clone();
    foo(&mut sm.x); //~ ERROR cannot borrow `sm.x` as mutable, as it is behind a `&` reference

    let x = X(Y);
    let xr = &x;
    let mut xm = xr.clone();
    bar(&mut xm); //~ ERROR cannot borrow data in a `&` reference as mutable
}
