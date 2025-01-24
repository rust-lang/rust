use super::*;

// The tests below compare the computed hashes to particular expected values
// in order to test that we produce the same results on different platforms,
// regardless of endianness and `usize` and `isize` size differences (this
// of course assumes we run these tests on platforms that differ in those
// ways). The expected values depend on the hashing algorithm used, so they
// need to be updated whenever StableHasher changes its hashing algorithm.

fn hash<T: HashStable<()>>(t: &T) -> Hash128 {
    let mut h = StableHasher::new();
    let ctx = &mut ();
    t.hash_stable(ctx, &mut h);
    h.finish()
}

// Check that bit set hash includes the domain size.
#[test]
fn test_hash_bit_set() {
    use rustc_index::bit_set::DenseBitSet;
    let a: DenseBitSet<usize> = DenseBitSet::new_empty(1);
    let b: DenseBitSet<usize> = DenseBitSet::new_empty(2);
    assert_ne!(a, b);
    assert_ne!(hash(&a), hash(&b));
}

// Check that bit matrix hash includes the matrix dimensions.
#[test]
fn test_hash_bit_matrix() {
    use rustc_index::bit_set::BitMatrix;
    let a: BitMatrix<usize, usize> = BitMatrix::new(1, 1);
    let b: BitMatrix<usize, usize> = BitMatrix::new(1, 2);
    assert_ne!(a, b);
    assert_ne!(hash(&a), hash(&b));
}

// Check that exchanging the value of two adjacent fields changes the hash.
#[test]
fn test_attribute_permutation() {
    macro_rules! test_type {
        ($ty: ty) => {{
            struct Foo {
                a: $ty,
                b: $ty,
            }

            impl<CTX> HashStable<CTX> for Foo {
                fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
                    self.a.hash_stable(hcx, hasher);
                    self.b.hash_stable(hcx, hasher);
                }
            }

            #[allow(overflowing_literals)]
            let mut item = Foo { a: 0xFF, b: 0xFF_FF };
            let hash_a = hash(&item);
            std::mem::swap(&mut item.a, &mut item.b);
            let hash_b = hash(&item);
            assert_ne!(
                hash_a,
                hash_b,
                "The hash stayed the same after values were swapped for type `{}`!",
                stringify!($ty)
            );
        }};
    }

    test_type!(u16);
    test_type!(u32);
    test_type!(u64);
    test_type!(u128);

    test_type!(i16);
    test_type!(i32);
    test_type!(i64);
    test_type!(i128);
}

// Check that the `isize` hashing optimization does not produce the same hash when permuting two
// values.
#[test]
fn test_isize_compression() {
    fn check_hash(a: u64, b: u64) {
        let hash_a = hash(&(a as isize, b as isize));
        let hash_b = hash(&(b as isize, a as isize));
        assert_ne!(
            hash_a, hash_b,
            "The hash stayed the same when permuting values `{a}` and `{b}`!",
        );
    }

    check_hash(0xAA, 0xAAAA);
    check_hash(0xFF, 0xFFFF);
    check_hash(0xAAAA, 0xAAAAAA);
    check_hash(0xAAAAAA, 0xAAAAAAAA);
    check_hash(0xFF, 0xFFFFFFFFFFFFFFFF);
    check_hash(u64::MAX /* -1 */, 1);
}
