pub static global: isize = 3;

static global0: isize = 4;

pub static global2: &'static isize = &global0;

pub fn verify_same(a: &'static isize) {
    let a = a as *const isize as usize;
    let b = &global as *const isize as usize;
    assert_eq!(a, b);
}

pub fn verify_same2(a: &'static isize) {
    let a = a as *const isize as usize;
    let b = global2 as *const isize as usize;
    assert_eq!(a, b);
}
