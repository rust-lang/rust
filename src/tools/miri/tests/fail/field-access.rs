fn main() {
    struct Foo { x: u8, y: u8 };

    let a = Foo { x: 0, y: 1 };

    let ap: *const Foo = &a as *const _;
    let xp = unsafe { std::ptr::addr_of!((*ap).x) };
    let yp = unsafe { std::ptr::addr_of!((*ap).y) };

    let xpp1 = unsafe { xp.offset(1) };
    assert_eq!(xpp1, yp);

    let bad_y = unsafe { *xpp1 }; //~ ERROR: only permits access to offsets 0..1
    println!("{}", bad_y);
}
