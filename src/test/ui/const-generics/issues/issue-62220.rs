// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

pub struct Vector<T, const N: usize>([T; N]);

pub type TruncatedVector<T, const N: usize> = Vector<T, { N - 1 }>;
//[min]~^ ERROR generic parameters may not be used in const operations

impl<T, const N: usize> Vector<T, { N }> {
    /// Drop the last component and return the vector with one fewer dimension.
    pub fn trunc(self) -> (TruncatedVector<T, { N }>, T) {
        //[full]~^ ERROR constant expression depends on a generic parameter
        unimplemented!()
    }
}

fn vec4<T>(a: T, b: T, c: T, d: T) -> Vector<T, 4> {
    Vector([a, b, c, d])
}

fn main() {
    let (_xyz, _w): (TruncatedVector<u32, 4>, u32) = vec4(0u32, 1, 2, 3).trunc();
}
