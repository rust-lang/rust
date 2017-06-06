#![recursion_limit="128"]

pub mod old {

}

pub mod new {
    macro_rules! blow {
        () => {};
        (_ $($rest:tt)*) => {
            pub mod a { blow!($($rest)*); }
            pub mod b { pub use super::a::*; }
        }
    }

    blow!(_ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _
          _ _ _ _ _ _ _ _);
}
