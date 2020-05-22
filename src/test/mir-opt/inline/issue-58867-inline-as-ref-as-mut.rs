// EMIT_MIR rustc.a.Inline.after.mir
pub fn a<T>(x: &mut [T]) -> &mut [T] {
    x.as_mut()
}

// EMIT_MIR rustc.b.Inline.after.mir
pub fn b<T>(x: &mut Box<T>) -> &mut T {
    x.as_mut()
}

// EMIT_MIR rustc.c.Inline.after.mir
pub fn c<T>(x: &[T]) -> &[T] {
    x.as_ref()
}

// EMIT_MIR rustc.d.Inline.after.mir
pub fn d<T>(x: &Box<T>) -> &T {
    x.as_ref()
}

fn main() {
    let mut boxed = Box::new(1);
    println!("{:?}", a(&mut [1]));
    println!("{:?}", b(&mut boxed));
    println!("{:?}", c(&[1]));
    println!("{:?}", d(&boxed));
}
