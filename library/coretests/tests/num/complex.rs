use core::num::{Complex, Wrapping};

#[test]
fn complex_addition() {
    let a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im));
    assert_eq!(a + b, b + a);
    assert_eq!(a + 8, Complex::new(a.re + 8, a.im));

    let a = Complex::new(Wrapping(1u8), Wrapping(2));
    let b = Complex::new(Wrapping(3u8), Wrapping(4));
    assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im));
    assert_eq!(a + b, b + a);
    let c = a + Wrapping(u8::MAX);
    assert_eq!(c, Complex::new(a.re + Wrapping(u8::MAX), a.im));
    assert_eq!(c.re.0, 1u8.wrapping_add(u8::MAX));

    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im));
    assert_eq!(a + b, b + a);
    assert_eq!(a + 8.0, Complex::new(a.re + 8.0, a.im));
}

#[test]
fn complex_subtraction() {
    let a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    assert_eq!(a - b, Complex::new(a.re - b.re, a.im - b.im));
    assert_eq!(a - 8, Complex::new(a.re - 8, a.im));

    let a = Complex::new(Wrapping(1u8), Wrapping(2));
    let b = Complex::new(Wrapping(3u8), Wrapping(4));
    assert_eq!(a - b, Complex::new(a.re - b.re, a.im - b.im));
    let c = a - Wrapping(u8::MAX);
    assert_eq!(c, Complex::new(a.re - Wrapping(u8::MAX), a.im));
    assert_eq!(c.re.0, 1u8.wrapping_sub(u8::MAX));

    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    assert_eq!(a - b, Complex::new(a.re - b.re, a.im - b.im));
    assert_eq!(a - 8.0, Complex::new(a.re - 8.0, a.im));
}
