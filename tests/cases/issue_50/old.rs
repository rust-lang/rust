pub trait TypedArrayElement {
    type Element;
}

pub struct CreateWith<'a, T: 'a>(&'a T);

pub fn create<T: TypedArrayElement>(_: CreateWith<T::Element>) { }

/* #[allow(dead_code)]
pub struct TypedArray<T: TypedArrayElement> {
    computed: T::Element,
}

impl<T: TypedArrayElement> TypedArray<T> {
    pub unsafe fn create(_: CreateWith<T::Element>) { }
} */
