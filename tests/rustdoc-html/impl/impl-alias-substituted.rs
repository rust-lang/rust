pub struct Matrix<T, const N: usize, const M: usize>([[T; N]; M]);

pub type Vector<T, const N: usize> = Matrix<T, N, 1>;

//@ has "impl_alias_substituted/struct.Matrix.html" '//*[@class="impl"]//h3[@class="code-header"]' \
//  "impl<T: Copy> Matrix<T, 3, 1>"
impl<T: Copy> Vector<T, 3> {
    pub fn test() {}
}
