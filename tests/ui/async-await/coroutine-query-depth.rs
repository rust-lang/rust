// Regression test for https://github.com/rust-lang/rust/issues/152942
//
// Query depth for computing layout of `deep()`'s coroutine body:
//
//   layout_of(coroutine)               depth 1
//     layout_of(S4)                    depth 2
//       layout_of(S3)                  depth 3
//         layout_of(S2)                depth 4
//           layout_of(S1)              depth 5
//             layout_of(S0)            depth 6
//               layout_of(u8)          depth 7
//
//   total: 7 < 8 -> won't overflow
//
// We used to have `MaybeUninit` wrap coroutine locals
// and that made query depth 10 which exceeds 8

//@ check-pass
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
