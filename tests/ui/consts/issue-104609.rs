fn foo() {
    oops;
    //~^ ERROR: cannot find value `oops` in this scope
}

unsafe fn bar() {
    std::mem::transmute::<_, *mut _>(1_u8);
    //~^ ERROR: type annotations needed
}

fn main() {}
