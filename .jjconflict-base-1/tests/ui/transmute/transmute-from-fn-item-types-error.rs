use std::mem;

unsafe fn foo() -> (i8, *const (), Option<fn()>) {
    let i = mem::transmute(bar);
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types


    let p = mem::transmute(foo);
    //~^ ERROR can't transmute zero-sized type


    let of = mem::transmute(main);
    //~^ ERROR can't transmute zero-sized type


    (i, p, of)
}

unsafe fn bar() {
    // Error as usual if the resulting type is not pointer-sized.
    mem::transmute::<_, u8>(main);
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types


    mem::transmute::<_, *mut ()>(foo);
    //~^ ERROR can't transmute zero-sized type


    mem::transmute::<_, fn()>(bar);
    //~^ ERROR can't transmute zero-sized type


    // No error if a coercion would otherwise occur.
    mem::transmute::<fn(), usize>(main);
}

unsafe fn baz() {
    mem::transmute::<_, *mut ()>(Some(foo));
    //~^ ERROR can't transmute zero-sized type


    mem::transmute::<_, fn()>(Some(bar));
    //~^ ERROR can't transmute zero-sized type


    mem::transmute::<_, Option<fn()>>(Some(baz));
    //~^ ERROR can't transmute zero-sized type


    // No error if a coercion would otherwise occur.
    mem::transmute::<Option<fn()>, usize>(Some(main));
}

fn main() {
    unsafe {
        foo();
        bar();
        baz();
    }
}
