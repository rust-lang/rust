const XY_1: i32 = 10;

fn main() {
    const XY_2: i32 = 20;
    let a = concat_idents!(X, Y_1); //~ ERROR `concat_idents` is not stable
    let b = concat_idents!(X, Y_2); //~ ERROR `concat_idents` is not stable
    assert_eq!(a, 10);
    assert_eq!(b, 20);
}
