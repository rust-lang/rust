
macro_rules! overly_complicated {
    {$fnname:ident, $arg:ident, $ty:ty, $body:block, $val:expr, $pat:pat, $res:path} =>
    {
        fn $fnname($arg: $ty) -> option<$ty> $body
        alt $fnname($val) {
          some($pat) => {
            $res
          }
          _ => { fail; }
        }
    }

}
fn main() {
    assert overly_complicated!(f, x, option<uint>, { return some(x); },
                               some(8u), some(y), y) == 8u

}