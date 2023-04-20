#[test]
fn bits_constant() {
    use crate::tagged_ptr::Tag;

    #[derive(Copy, Clone)]
    struct Unit;
    impl_tag! { impl Tag for Unit; Unit <=> 0, }
    assert_eq!(Unit::BITS, 0);

    #[derive(Copy, Clone)]
    struct Unit1;
    impl_tag! { impl Tag for Unit1; Unit1 <=> 1, }
    assert_eq!(Unit1::BITS, 1);

    #[derive(Copy, Clone)]
    struct Unit2;
    impl_tag! { impl Tag for Unit2; Unit2 <=> 0b10, }
    assert_eq!(Unit2::BITS, 2);

    #[derive(Copy, Clone)]
    struct Unit3;
    impl_tag! { impl Tag for Unit3; Unit3 <=> 0b100, }
    assert_eq!(Unit3::BITS, 3);

    #[derive(Copy, Clone)]
    enum Enum {
        A,
        B,
        C,
    }
    impl_tag! { impl Tag for Enum; Enum::A <=> 0b1, Enum::B <=> 0b1000, Enum::C <=> 0b10, }
    assert_eq!(Enum::BITS, 4);
}
