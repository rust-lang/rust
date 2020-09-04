// check-pass
fn foo() -> usize { 42 }
fn bar(x: usize) -> usize { x }
fn baz(x: usize, y: usize) -> usize { x + y }

fn main() {
    println!("{:p}", &foo);
    //~^ WARN cast `foo` with `as *const fn() -> _` to use it as a pointer
    println!("{:p}", &bar);
    //~^ WARN cast `bar` with `as *const fn(_) -> _` to use it as a pointer
    println!("{:p}", &baz);
    //~^ WARN cast `baz` with `as *const fn(_, _) -> _` to use it as a pointer

    //should not produce any warnings
    println!("{:p}", foo as *const fn() -> usize);
    println!("{:p}", bar as *const fn(usize) -> usize);
    println!("{:p}", baz as *const fn(usize, usize) -> usize);

    //should not produce any warnings
    let fn_thing = foo;
    println!("{:p}", &fn_thing);
}
