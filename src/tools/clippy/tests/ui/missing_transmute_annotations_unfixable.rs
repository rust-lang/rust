//@no-rustfix

fn issue14984() {
    async fn e() {}
    async fn x() -> u32 {
        0
    }
    async fn y() -> f32 {
        0.0
    };
    let mut yy = unsafe { std::ptr::read(&y()) };
    yy = unsafe { std::mem::transmute(std::ptr::read(&x())) };
    //~^ missing_transmute_annotations

    let mut zz = 0u8;
    zz = unsafe { std::mem::transmute(std::ptr::read(&x())) };
    //~^ missing_transmute_annotations

    yy = unsafe { std::mem::transmute(zz) };
    //~^ missing_transmute_annotations

    fn a() -> impl Sized {
        0u32
    }

    let mut b: f32 = 0.0;
    b = unsafe { std::mem::transmute(a()) };
    //~^ missing_transmute_annotations
}
