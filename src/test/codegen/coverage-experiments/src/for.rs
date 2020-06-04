#[inline(always)]
pub fn __incr_cov<T>(_region_loc: &str, /*index: u32,*/ result: T) -> T {
    result
}

fn main() {
    for countdown in __incr_cov("start", 10..0) {
        let _ = countdown;
        __incr_cov("top of for", ());
    }
}

// LOWERED TO HIR:
//
// fn main() {
//   {
//       let _t =
//           match ::std::iter::IntoIterator::into_iter(__incr_cov("start",
//                                                                 ::std::ops::Range{start:
//                                                                                       10,
//                                                                                   end:
//                                                                                       0,}))
//               {
//               mut iter =>
//               loop  {
//                   let mut __next;
//                   match ::std::iter::Iterator::next(&mut iter) {
//                       ::std::option::Option::Some(val) =>
//                       __next = val,
//                       ::std::option::Option::None => break ,
//                   }
//                   let countdown = __next;
//                   {
//                       let _ = countdown;
//                       __incr_cov("top of for", ());
//                   }
//               },
//           };
//       _t
//   }
// }