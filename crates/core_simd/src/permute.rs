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

            /// Reverse the order of the lanes in the vector.
            #[inline]
            pub fn reverse(self) -> Self {
                const fn idx() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        idx[i] = ($n - i - 1) as u32;
                        i += 1;
                    }
                    idx
                }
                self.shuffle::<{ idx() }>(self)
            }

            /// Interleave two vectors.
            ///
            /// The even lanes of the first result contain the lower half of `self`, and the odd
            /// lanes contain the lower half of `other`.
            ///
            /// The even lanes of the second result contain the upper half of `self`, and the odd
            /// lanes contain the upper half of `other`.
            #[inline]
            pub fn interleave(self, other: Self) -> (Self, Self) {
                const fn lo() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        let offset = i / 2;
                        idx[i] = if i % 2 == 0 {
                            offset
                        } else {
                            $n + offset
                        } as u32;
                        i += 1;
                    }
                    idx
                }
                const fn hi() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        let offset = ($n + i) / 2;
                        idx[i] = if i % 2 == 0 {
                            offset
                        } else {
                            $n + offset
                        } as u32;
                        i += 1;
                    }
                    idx
                }
                (self.shuffle::<{ lo() }>(other), self.shuffle::<{ hi() }>(other))
            }

            /// Deinterleave two vectors.
            ///
            /// The first result contains the even lanes of `self` and `other` concatenated.
            ///
            /// The second result contains the odd lanes of `self` and `other` concatenated.
            #[inline]
            pub fn deinterleave(self, other: Self) -> (Self, Self) {
                const fn even() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        idx[i] = 2 * i as u32;
                        i += 1;
                    }
                    idx
                }
                const fn odd() -> [u32; $n] {
                    let mut idx = [0u32; $n];
                    let mut i = 0;
                    while i < $n {
                        idx[i] = 1 + 2 * i as u32;
                        i += 1;
                    }
                    idx
                }
                (self.shuffle::<{ even() }>(other), self.shuffle::<{ odd() }>(other))
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
