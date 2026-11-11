//@aux-build:proc_macros.rs
#![warn(clippy::manual_assert_eq)]
#![expect(clippy::assertions_on_constants, clippy::eq_op)]

fn main() {
    let a = "a";
    assert!(a == "a".to_ascii_lowercase());
    //~^ manual_assert_eq
    assert!(a != "a".to_ascii_uppercase());
    //~^ manual_assert_eq
    debug_assert!(a == "a".to_ascii_lowercase());
    //~^ manual_assert_eq
    debug_assert!(a != "a".to_ascii_uppercase());
    //~^ manual_assert_eq

    // macros
    let v = vec![];
    assert!(v == vec![1, 2, 3]);
    //~^ manual_assert_eq
    assert!(vec![1, 2, 3] == v);
    //~^ manual_assert_eq
    assert!(vec![1] == vec![1, 2, 3]);
    //~^ manual_assert_eq

    // Don't lint: has assert message
    assert!(a == "a".to_ascii_lowercase(), "{a}");
    assert!(a == "a".to_ascii_lowercase(), "a==a");
    assert!(a == "a".to_ascii_lowercase(), "{a}==a");
    assert!(a != "a".to_ascii_uppercase(), "a!=A");
    debug_assert!(a == "a".to_ascii_lowercase(), "a==a");
    debug_assert!(a != "a".to_ascii_uppercase(), "a!=A");

    // Don't lint: `!=`, and at least one of the sides is a constant value
    assert!(a != "A");
    assert!("A" != a);
    assert!("A" != "A");

    // Don't lint: comparison of ptrs
    fn cmp_ptrs(a: *const u8, b: *const u8) {
        assert!(a == b);
    }

    // Don't lint: one of the sides isn't `Debug`
    {
        #[derive(PartialEq)]
        struct NotDebug;

        #[derive(PartialEq)]
        struct NotDebug2;

        impl PartialEq<NotDebug2> for NotDebug {
            fn eq(&self, other: &NotDebug2) -> bool {
                unimplemented!()
            }
        }
        impl PartialEq<NotDebug> for NotDebug2 {
            fn eq(&self, other: &NotDebug) -> bool {
                unimplemented!()
            }
        }

        #[derive(Debug)]
        struct IsDebug;

        impl PartialEq<IsDebug> for NotDebug {
            fn eq(&self, other: &IsDebug) -> bool {
                unimplemented!()
            }
        }
        impl PartialEq<NotDebug> for IsDebug {
            fn eq(&self, other: &NotDebug) -> bool {
                unimplemented!()
            }
        }

        let nd = NotDebug;
        assert!(nd == nd);

        let nd2 = NotDebug2;
        assert!(nd == nd2);
        assert!(nd2 == nd);

        let id = IsDebug;
        assert!(id == nd);
        assert!(nd == id);
    }

    // Don't lint: byte buffers can contain too much data for useful debug output
    {
        use std::borrow::Cow;
        use std::ops::Deref;
        use std::rc::Rc;
        use std::sync::Arc;

        #[derive(Debug, PartialEq)]
        struct ByteBuf(Vec<u8>);

        impl Deref for ByteBuf {
            type Target = [u8];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        let vec = vec![1_u8];
        assert!(vec == vec![1]);

        let slice: &[u8] = &[1];
        let expected_slice: &[u8] = &[1];
        assert!(slice == expected_slice);

        let boxed: Box<[u8]> = Box::new([1]);
        assert!(boxed == Box::new([1]));

        let cow: Cow<'_, [u8]> = Cow::Borrowed(&[1]);
        assert!(cow == Cow::Borrowed(&[1]));

        let rc: Rc<[u8]> = Rc::new([1]);
        assert!(rc == Rc::new([1]));

        let arc: Arc<[u8]> = Arc::new([1]);
        assert!(arc == Arc::new([1]));

        let custom = ByteBuf(vec![1]);
        assert!(custom == ByteBuf(vec![1]));

        #[derive(Debug, PartialEq)]
        struct InnerPiece([u8; 1024]);

        #[derive(Debug, PartialEq)]
        struct Piece(InnerPiece);

        impl Deref for Piece {
            type Target = InnerPiece;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl AsRef<[u8]> for Piece {
            fn as_ref(&self) -> &[u8] {
                &self.0.0
            }
        }

        let piece = Piece(InnerPiece([0; 1024]));
        assert!(piece == Piece(InnerPiece([0; 1024])));

        #[derive(Debug, PartialEq)]
        struct Grow<T>(std::marker::PhantomData<T>);

        impl<T> Deref for Grow<T> {
            type Target = Grow<(T,)>;

            fn deref(&self) -> &Self::Target {
                unreachable!()
            }
        }

        assert!(Grow::<u8>(std::marker::PhantomData) == Grow(std::marker::PhantomData));
        //~^ manual_assert_eq

        #[derive(Debug, PartialEq)]
        struct SelfDerefer;

        impl Deref for SelfDerefer {
            type Target = Self;

            fn deref(&self) -> &Self::Target {
                self
            }
        }

        assert!(SelfDerefer == SelfDerefer);
        //~^ manual_assert_eq
    }

    // Don't lint: in const context
    const {
        assert!(5 == 2 + 3);
    }

    // Don't lint: in external macro
    {
        // NOTE: this only works because `root_macro_call_first_node` returns `external!`,
        // which then gets rejected by the macro name check
        proc_macros::external!(assert!('a' == 'b'));
        proc_macros::external!({
            let some_padding_before = 'a';
            assert!('a' == 'b');
            let some_padding_after = 'b';
        });

        // .. which also means that the following is _technically_ a FN -- but surely no one would write
        // code like this (diverging/unit expression as a child expression of a macro call)
        vec![(), assert!('a' == 'b'), ()];
    }
}

// Don't lint: in const context
const _: () = {
    assert!(8 == (7 + 1));
};
