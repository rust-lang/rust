/// Implements the `+` operator for concatenating two `char`s together.
///
/// This operation results into a new `String` being allocated with copies of the two `char`s in
/// the respective order.
///
/// # Examples
///
/// ```
/// let a = '🎈';
/// let b = '🎉';
/// let c = a + b;
/// // `c` is a newly allocated `String`
/// ```
#[stable(feature = "add_chars", since = "1.41.0")]
impl Add<char> for char {
    type Output = String;

    #[inline]
    fn add(self, other: char) -> String {
        let mut r = String::new();
        r.push(self);
        r.push(other);
        
        r
    }
}
