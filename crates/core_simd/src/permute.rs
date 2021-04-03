macro_rules! impl_shuffle_lane {
    { $name:ident, $fn:ident, $n:literal } => {
        impl $name<$n> {
            /// A const SIMD shuffle that takes 2 SIMD vectors and produces another vector, using
            /// the indices in the const parameter. The first or "self" vector will have its lanes
            /// indexed from 0, and the second vector will have its first lane indexed at $n.
            /// Indices must be in-bounds of either vector at compile time.
            ///
            /// Some SIMD shuffle instructions can be quite slow, so avoiding them by loading data
            /// into the desired patterns in advance is preferred, but shuffles are still faster
            /// than storing and reloading from memory.
            #[inline]
            pub fn shuffle<const IDX: [u32; $n]>(self, second: Self) -> Self {
                unsafe { crate::intrinsics::$fn(self, second, IDX) }
            }
        }
    }
}

macro_rules! impl_shuffle_2pow_lanes {
    { $name:ident } => {
        impl_shuffle_lane!{ $name, simd_shuffle2, 2 }
        impl_shuffle_lane!{ $name, simd_shuffle4, 4 }
        impl_shuffle_lane!{ $name, simd_shuffle8, 8 }
        impl_shuffle_lane!{ $name, simd_shuffle16, 16 }
        impl_shuffle_lane!{ $name, simd_shuffle32, 32 }
    }
}
