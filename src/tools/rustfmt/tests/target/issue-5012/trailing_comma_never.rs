// rustfmt-trailing_comma: Never

pub struct Matrix<T, const R: usize, const C: usize>
where
    [T; R * C]:
{
    contents: [T; R * C]
}
