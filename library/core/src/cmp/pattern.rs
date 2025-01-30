use crate::cmp::BytewiseEq;
use crate::intrinsics::compare_bytes;
use crate::mem;

#[lang = "PatternConstEq"]
#[const_trait]
trait PatternConstEq<Rhs = Self>
where
    Rhs: ?Sized,
{
    #[allow(dead_code)]
    fn eq(&self, other: &Rhs) -> bool;
}

macro_rules ! impl_for_primitive {
    ($($t:ty),*) => {
        $(
            impl const PatternConstEq for $t {
                fn eq(&self, other: &Self) -> bool {
                    *self == *other
                }
            }
        )*
    };
}

impl_for_primitive! {
    bool, char,
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64
}

impl<T> const PatternConstEq for [T]
where
    T: ~const PatternConstEq,
{
    default fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let mut i = 0;

        while i < self.len() {
            if <T as PatternConstEq>::eq(&self[i], &other[i]) == false {
                return false;
            }

            i += 1;
        }

        true
    }
}

#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
impl<T> const PatternConstEq for [T]
where
    T: ~const PatternConstEq + BytewiseEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // SAFETY: `self` and `other` are references and are thus guaranteed to be valid.
        // The two slices have been checked to have the same size above.
        unsafe {
            let size = mem::size_of_val(self);
            compare_bytes(self.as_ptr() as *const u8, other.as_ptr() as *const u8, size) == 0
        }
    }
}
