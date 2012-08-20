
macro_rules! overly_complicated (
    ($fnname:ident, $arg:ident, $ty:ty, $body:block, $val:expr, $pat:pat, $res:path) =>
    {
        fn $fnname($arg: $ty) -> Option<$ty> $body
        match $fnname($val) {
          Some($pat) => {
            $res
          }
          _ => { fail; }
        }
    }

)
fn main() {
    assert overly_complicated!(f, x, Option<uint>, { return Some(x); },
                               Some(8u), Some(y), y) == 8u

}