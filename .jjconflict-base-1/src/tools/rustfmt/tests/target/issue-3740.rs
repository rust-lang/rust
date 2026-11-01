impl<T, const SIZE: usize> IntoNormalized for Vector<T, { SIZE }>
where
    Vector<T, { SIZE }>: Div<Vector<T, { SIZE }>>,
    for<'a> &'a Vector<T, { SIZE }>: IntoLength<Output = T>,
{
    type Output = Vector<T, { SIZE }>;
    fn into_normalized(self) -> Self::Output {}
}
