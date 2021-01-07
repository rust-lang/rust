macro_rules! int_tests {
    { $vector:ident, $scalar:ident } => {
        #[cfg(test)]
        mod $vector {
            use super::*;
            use helpers::lanewise::*;

            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen_test::*;

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_test_configure!(run_in_browser);

            // TODO impl this as an associated fn on vectors
            fn from_slice(slice: &[$scalar]) -> core_simd::$vector {
                let mut value = core_simd::$vector::default();
                let value_slice: &mut [_] = value.as_mut();
                value_slice.copy_from_slice(&slice[0..value_slice.len()]);
                value
            }

            const A: [$scalar; 64] = [
                7, 7, 7, 7, -7, -7, -7, -7,
                6, 6, 6, 6, -6, -6, -6, -6,
                5, 5, 5, 5, -5, -5, -5, -5,
                4, 4, 4, 4, -4, -4, -4, -4,
                3, 3, 3, 3, -3, -3, -3, -3,
                2, 2, 2, 2, -2, -2, -2, -2,
                1, 1, 1, 1, -1, -1, -1, -1,
                0, 0, 0, 0,  0,  0,  0,  0,
            ];
            const B: [$scalar; 64] = [
                 1,  2,  3,  4,  5,  6,  7,  8,
                 1,  2,  3,  4,  5,  6,  7,  8,
                 1,  2,  3,  4,  5,  6,  7,  8,
                 1,  2,  3,  4,  5,  6,  7,  8,
                 -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,
                 -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,
                 -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,
                 -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,
            ];

            #[test]
            #[should_panic]
            fn div_min_panics() {
                let a = from_slice(&vec![$scalar::MIN; 64]);
                let b = from_slice(&vec![-1; 64]);
                let _ = a / b;
            }

            #[test]
            #[should_panic]
            fn div_by_all_zeros_panics() {
                let a = from_slice(&A);
                let b = from_slice(&vec![0 ; 64]);
                let _ = a / b;
            }

            #[test]
            #[should_panic]
            fn div_by_one_zero_panics() {
                let a = from_slice(&A);
                let mut b = from_slice(&B);
                b[0] = 0 as _;
                let _ = a / b;
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn div_min_neg_one_no_panic() {
                let a = from_slice(&A);
                let b = from_slice(&vec![-1; 64]);
                let _ = a / b;
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn rem_min_neg_one_no_panic() {
                let a = from_slice(&A);
                let b = from_slice(&vec![-1; 64]);
                let _ = a % b;
            }

            #[test]
            #[should_panic]
            fn rem_min_panic() {
                let a = from_slice(&vec![$scalar::MIN; 64]);
                let b = from_slice(&vec![-1 ; 64]);
                let _ = a % b;
            }

            #[test]
            #[should_panic]
            fn rem_min_zero_panic() {
                let a = from_slice(&A);
                let b = from_slice(&vec![0 ; 64]);
                let _ = a % b;
            }
        }
    }
}
