//! Implements formating traits.
#![allow(unused)]

macro_rules! impl_hex_fmt {
    ($id:ident, $elem_ty:ident) => {
        impl ::fmt::LowerHex for $id {
            fn fmt(&self, f: &mut ::fmt::Formatter)
                   -> ::fmt::Result {
                use ::mem;
                write!(f, "{}(", stringify!($id))?;
                let n = mem::size_of_val(self)
                    / mem::size_of::<$elem_ty>();
                for i in 0..n {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.extract(i).fmt(f)?;
                }
                write!(f, ")")
            }
        }
        impl ::fmt::UpperHex for $id {
            fn fmt(&self, f: &mut ::fmt::Formatter)
                   -> ::fmt::Result {
                write!(f, "{}(", stringify!($id))?;
                for i in 0..$id::lanes() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.extract(i).fmt(f)?;
                }
                write!(f, ")")
            }
        }
        impl ::fmt::Octal for $id {
            fn fmt(&self, f: &mut ::fmt::Formatter)
                   -> ::fmt::Result {
                write!(f, "{}(", stringify!($id))?;
                for i in 0..$id::lanes() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.extract(i).fmt(f)?;
                }
                write!(f, ")")
            }
        }
        impl ::fmt::Binary for $id {
            fn fmt(&self, f: &mut ::fmt::Formatter)
                   -> ::fmt::Result {
                write!(f, "{}(", stringify!($id))?;
                for i in 0..$id::lanes() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.extract(i).fmt(f)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_hex_fmt_impl {
    ($id:ident, $elem_ty:ident, $($values:expr),+) => {
        #[test]
        fn hex_fmt() {
            use ::std::prelude::v1::*;
            use ::coresimd::simd::$id;
            for &i in [$($values),+].iter() {
                let vec = $id::splat(i as $elem_ty);

                let s = format!("{:#x}", vec);
                let beg = format!("{}(", stringify!($id));
                assert!(s.starts_with(&beg));
                assert!(s.ends_with(")"));
                let s: Vec<String> = s.replace(&beg, "").replace(")", "").split(",")
                    .map(|v| v.trim().to_string()).collect();
                assert_eq!(s.len(), $id::lanes());
                for (index, ss) in s.into_iter().enumerate() {
                    assert_eq!(ss, format!("{:#x}", vec.extract(index)));
                }
            }
        }
        #[test]
        fn upper_hex_fmt() {
            use ::std::prelude::v1::*;
            use ::coresimd::simd::$id;
            for &i in [$($values),+].iter() {
                let vec = $id::splat(i as $elem_ty);

                let s = format!("{:#X}", vec);
                let beg = format!("{}(", stringify!($id));
                assert!(s.starts_with(&beg));
                assert!(s.ends_with(")"));
                let s: Vec<String> = s.replace(&beg, "").replace(")", "").split(",")
                    .map(|v| v.trim().to_string()).collect();
                assert_eq!(s.len(), $id::lanes());
                for (index, ss) in s.into_iter().enumerate() {
                    assert_eq!(ss, format!("{:#X}", vec.extract(index)));
                }
            }
        }
        #[test]
        fn octal_fmt() {
            use ::std::prelude::v1::*;
            use ::coresimd::simd::$id;
            for &i in [$($values),+].iter() {
                let vec = $id::splat(i as $elem_ty);

                let s = format!("{:#o}", vec);
                let beg = format!("{}(", stringify!($id));
                assert!(s.starts_with(&beg));
                assert!(s.ends_with(")"));
                let s: Vec<String> = s.replace(&beg, "").replace(")", "").split(",")
                    .map(|v| v.trim().to_string()).collect();
                assert_eq!(s.len(), $id::lanes());
                for (index, ss) in s.into_iter().enumerate() {
                    assert_eq!(ss, format!("{:#o}", vec.extract(index)));
                }
            }
        }
        #[test]
        fn binary_fmt() {
            use ::std::prelude::v1::*;
            use ::coresimd::simd::$id;
            for &i in [$($values),+].iter() {
                let vec = $id::splat(i as $elem_ty);

                let s = format!("{:#b}", vec);
                let beg = format!("{}(", stringify!($id));
                assert!(s.starts_with(&beg));
                assert!(s.ends_with(")"));
                let s: Vec<String> = s.replace(&beg, "").replace(")", "").split(",")
                    .map(|v| v.trim().to_string()).collect();
                assert_eq!(s.len(), $id::lanes());
                for (index, ss) in s.into_iter().enumerate() {
                    assert_eq!(ss, format!("{:#b}", vec.extract(index)));
                }
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_hex_fmt {
    ($id:ident, $elem_ty:ident) => {
        test_hex_fmt_impl!($id, $elem_ty, 0 as $elem_ty, !(0 as $elem_ty), (1 as $elem_ty));
    }
}
