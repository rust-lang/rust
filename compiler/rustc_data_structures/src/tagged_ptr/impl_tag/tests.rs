#[test]
fn bits_constant() {
    use crate::tagged_ptr::Tag;

    #[derive(Copy, Clone)]
    struct Unit;
    impl_tag! { impl Tag for Unit; Unit, }
    assert_eq!(Unit::BITS, 0);

    #[derive(Copy, Clone)]
    enum Enum3 {
        A,
        B,
        C,
    }
    impl_tag! { impl Tag for Enum3; Enum3::A, Enum3::B, Enum3::C, }
    assert_eq!(Enum3::BITS, 2);

    #[derive(Copy, Clone)]
    struct Eight(bool, bool, bool);
    impl_tag! {
        impl Tag for Eight;
        Eight { 0: true,  1: true,  2: true  },
        Eight { 0: true,  1: true,  2: false },
        Eight { 0: true,  1: false, 2: true  },
        Eight { 0: true,  1: false, 2: false },
        Eight { 0: false, 1: true,  2: true  },
        Eight { 0: false, 1: true,  2: false },
        Eight { 0: false, 1: false, 2: true  },
        Eight { 0: false, 1: false, 2: false },
    }

    assert_eq!(Eight::BITS, 3);
}
