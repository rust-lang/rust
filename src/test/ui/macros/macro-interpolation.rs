// run-pass

macro_rules! overly_complicated {
    ($fnname:ident, $arg:ident, $ty:ty, $body:block, $val:expr, $pat:pat, $res:path) =>
    ({
        fn $fnname($arg: $ty) -> Option<$ty> $body
        match $fnname($val) {
          Some($pat) => {
            $res
          }
          _ => { panic!(); }
        }
    })

}

macro_rules! qpath {
    (<$type:ty as $trait:path>::$name:ident) => {
        <$type as $trait>::$name
    };
}

pub fn main() {
    let _: qpath!(<str as ToOwned>::Owned);

    assert!(overly_complicated!(f, x, Option<usize>, { return Some(x); },
                               Some(8), Some(y), y) == 8)
}
