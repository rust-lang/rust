pub static V: &u32 = &X;
pub static F: fn() = f;
pub static G: fn() = G0;
pub static H: &(dyn Fn() + Sync) = &h;
pub static I: fn() = Helper(j).mk();
pub static K: fn() -> fn() = {
    #[inline(never)]
    fn k() {}
    #[inline(always)]
    || -> fn() { k }
};

static X: u32 = 42;
static G0: fn() = g;

pub fn v() -> *const u32 {
    V
}

fn f() {}

fn g() {}

fn h() {}

#[derive(Copy, Clone)]
struct Helper<T: Copy>(T);

impl<T: Copy + FnOnce()> Helper<T> {
    const fn mk(self) -> fn() {
        i::<T>
    }
}

fn i<T: FnOnce()>() {
    assert_eq!(std::mem::size_of::<T>(), 0);
    // unsafe to work around the lack of a `Default` impl for function items
    unsafe { (std::mem::transmute_copy::<(), T>(&()))() }
}

fn j() {}
