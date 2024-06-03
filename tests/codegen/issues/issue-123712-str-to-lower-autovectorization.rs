//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

/// Ensure that the ascii-prefix loop for `str::to_lowercase` and `str::to_uppercase` uses vector
/// instructions. Since these methods do not get inlined, the relevant code is duplicated here and
/// should be updated when the implementation changes.
// CHECK-LABEL: @lower_while_ascii
// CHECK: [[A:%[0-9]]] = load <16 x i8>
// CHECK-NEXT: [[B:%[0-9]]] = icmp slt <16 x i8> [[A]], zeroinitializer
// CHECK-NEXT: [[C:%[0-9]]] = bitcast <16 x i1> [[B]] to i16
#[no_mangle]
pub fn lower_while_ascii(mut input: &[u8], mut output: &mut [u8]) -> usize {
    // process the input in chunks to enable auto-vectorization
    const USIZE_SIZE: usize = core::mem::size_of::<usize>();
    const MAGIC_UNROLL: usize = 2;
    const N: usize = USIZE_SIZE * MAGIC_UNROLL;

    output = &mut output[..input.len()];

    let mut ascii_prefix_len = 0_usize;
    let mut is_ascii = [false; N];

    while input.len() >= N {
        let chunk = unsafe { input.get_unchecked(..N) };
        let out_chunk = unsafe { output.get_unchecked_mut(..N) };

        for j in 0..N {
            is_ascii[j] = chunk[j] <= 127;
        }

        // auto-vectorization for this check is a bit fragile,
        // sum and comparing against the chunk size gives the best result,
        // specifically a pmovmsk instruction on x86.
        if is_ascii.iter().map(|x| *x as u8).sum::<u8>() as usize != N {
            break;
        }

        for j in 0..N {
            out_chunk[j] = chunk[j].to_ascii_lowercase();
        }

        ascii_prefix_len += N;
        input = unsafe { input.get_unchecked(N..) };
        output = unsafe { output.get_unchecked_mut(N..) };
    }

    ascii_prefix_len
}
