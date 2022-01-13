use super::*;

impl<T> ThinVec<T> {
    fn into_vec(self) -> Vec<T> {
        self.into()
    }
}

#[test]
fn test_from_iterator() {
    assert_eq!(std::iter::empty().collect::<ThinVec<String>>().into_vec(), Vec::<String>::new());
    assert_eq!(std::iter::once(42).collect::<ThinVec<_>>().into_vec(), vec![42]);
    assert_eq!([1, 2].into_iter().collect::<ThinVec<_>>().into_vec(), vec![1, 2]);
    assert_eq!([1, 2, 3].into_iter().collect::<ThinVec<_>>().into_vec(), vec![1, 2, 3]);
}

#[test]
fn test_into_iterator_owned() {
    assert_eq!(ThinVec::new().into_iter().collect::<Vec<String>>(), Vec::<String>::new());
    assert_eq!(ThinVec::from(vec![1]).into_iter().collect::<Vec<_>>(), vec![1]);
    assert_eq!(ThinVec::from(vec![1, 2]).into_iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(ThinVec::from(vec![1, 2, 3]).into_iter().collect::<Vec<_>>(), vec![1, 2, 3]);
}

#[test]
fn test_into_iterator_ref() {
    assert_eq!(ThinVec::new().iter().collect::<Vec<&String>>(), Vec::<&String>::new());
    assert_eq!(ThinVec::from(vec![1]).iter().collect::<Vec<_>>(), vec![&1]);
    assert_eq!(ThinVec::from(vec![1, 2]).iter().collect::<Vec<_>>(), vec![&1, &2]);
    assert_eq!(ThinVec::from(vec![1, 2, 3]).iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
}

#[test]
fn test_into_iterator_ref_mut() {
    assert_eq!(ThinVec::new().iter_mut().collect::<Vec<&mut String>>(), Vec::<&mut String>::new());
    assert_eq!(ThinVec::from(vec![1]).iter_mut().collect::<Vec<_>>(), vec![&mut 1]);
    assert_eq!(ThinVec::from(vec![1, 2]).iter_mut().collect::<Vec<_>>(), vec![&mut 1, &mut 2]);
    assert_eq!(
        ThinVec::from(vec![1, 2, 3]).iter_mut().collect::<Vec<_>>(),
        vec![&mut 1, &mut 2, &mut 3],
    );
}
