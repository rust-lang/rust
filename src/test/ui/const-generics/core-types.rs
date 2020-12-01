// Check that all types allowed with `min_const_generics` work.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct A<const N: u8>;
struct B<const N: u16>;
struct C<const N: u32>;
struct D<const N: u64>;
struct E<const N: u128>;
struct F<const N: usize>;
struct G<const N: i8>;
struct H<const N: i16>;
struct I<const N: i32>;
struct J<const N: i64>;
struct K<const N: i128>;
struct L<const N: isize>;
struct M<const N: char>;
struct N<const N: bool>;

fn main() {
    let _ = A::<{u8::MIN}>;
    let _ = A::<{u8::MAX}>;
    let _ = B::<{u16::MIN}>;
    let _ = B::<{u16::MAX}>;
    let _ = C::<{u32::MIN}>;
    let _ = C::<{u32::MAX}>;
    let _ = D::<{u64::MIN}>;
    let _ = D::<{u64::MAX}>;
    let _ = E::<{u128::MIN}>;
    let _ = E::<{u128::MAX}>;
    let _ = F::<{usize::MIN}>;
    let _ = F::<{usize::MAX}>;
    let _ = G::<{i8::MIN}>;
    let _ = G::<{i8::MAX}>;
    let _ = H::<{i16::MIN}>;
    let _ = H::<{i16::MAX}>;
    let _ = I::<{i32::MIN}>;
    let _ = I::<{i32::MAX}>;
    let _ = J::<{i64::MIN}>;
    let _ = J::<{i64::MAX}>;
    let _ = K::<{i128::MIN}>;
    let _ = K::<{i128::MAX}>;
    let _ = L::<{isize::MIN}>;
    let _ = L::<{isize::MAX}>;
    let _ = M::<'A'>;
    let _ = N::<true>;
}
