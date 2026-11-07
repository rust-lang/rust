//@ check-pass

// Previously the `CONST_EVALUATABLE_UNCHECKED` FCW would fire on const evaluation of
// associated consts. This is unnecessary as the FCW only needs to apply for repeat expr
// counts which are anon consts with generic parameters provided. #140447

pub struct Foo<const N: usize>;

impl<const N: usize> Foo<N> {
    const UNUSED_PARAM: usize = {
        let _: [(); N];
        3
    };

    pub fn bar() {
        match 1 {
            Self::UNUSED_PARAM => (),
            _ => (),
        }
    }
}

fn main() {}
