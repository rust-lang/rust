#![feature(box_patterns)]
#![feature(unsized_fn_params)]

#[allow(dead_code)]
fn f1(box box _b: Box<Box<[u8]>>) {}
//~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]

fn f2((_x, _y): (i32, [i32])) {}
//~^ ERROR: the size for values of type `[i32]` cannot be known at compilation time [E0277]

fn main() {
    let foo: Box<[u8]> = Box::new(*b"foo");
    let _foo: [u8] = *foo;
    //~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]
}
