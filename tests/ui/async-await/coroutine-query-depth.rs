//~ ERROR: queries overflow the depth limit!

// Regression test for https://github.com/rust-lang/rust/issues/152942
//
// When computing coroutine layout,
// each saved local is wrapped in `MaybeUninit<T>`,
// which chains through `ManuallyDrop<T>`
// and `MaybeDangling<T>` before reaching `T`.
// This adds 3 extra layout_of queries per field.
//
// Query depth for computing layout of `deep()`'s coroutine body
// (before fix, with recursion_limit = 8):
//
//   layout_of(coroutine)               depth 1
//     layout_of(MaybeUninit<S4>)       depth 2  <- wrapping adds
//       layout_of(ManuallyDrop<S4>)    depth 3  <- 3 extra
//         layout_of(MaybeDangling<S4>) depth 4  <- queries
//           layout_of(S4)              depth 5
//             layout_of(S3)            depth 6
//               layout_of(S2)          depth 7
//                 layout_of(S1)        depth 8
//                   layout_of(S0)      depth 9
//                     layout_of(u8)    depth 10
//
//   total: 10 > 8 -> overflow!

//@ check-fail
//@ edition: 2024

#![recursion_limit = "8"]

struct S0(u8);
struct S1(S0);
struct S2(S1);
struct S3(S2);
struct S4(S3);

async fn yield_now() {}

fn use_val<T>(_: &T) {}

async fn deep() {
    let x = S4(S3(S2(S1(S0(0)))));
    yield_now().await;
    use_val(&x);
}

fn main() {
    std::mem::size_of_val(&deep());
}
