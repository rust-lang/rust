#[rustc_const_unstable(feature = "const_unicode_case_lookup", issue = "101400")]
#[inline(always)]
const fn bitset_search<
    const N: usize,
    const CHUNK_SIZE: usize,
    const N1: usize,
    const CANONICAL: usize,
    const CANONICALIZED: usize,
>(
    needle: u32,
    chunk_idx_map: &[u8; N],
    bitset_chunk_idx: &[[u8; CHUNK_SIZE]; N1],
    bitset_canonical: &[u64; CANONICAL],
    bitset_canonicalized: &[(u8, u8); CANONICALIZED],
) -> bool {
    let bucket_idx = (needle / 64) as usize;
    let chunk_map_idx = bucket_idx / CHUNK_SIZE;
    let chunk_piece = bucket_idx % CHUNK_SIZE;
    // FIXME: const-hack: Revert to `slice::get` after `const_slice_index`
    // feature stabilizes.
    let chunk_idx = if chunk_map_idx < chunk_idx_map.len() {
        chunk_idx_map[chunk_map_idx]
    } else {
        return false;
    };
    let idx = bitset_chunk_idx[chunk_idx as usize][chunk_piece] as usize;
    // FIXME: const-hack: Revert to `slice::get` after `const_slice_index`
    // feature stabilizes.
    let word = if idx < bitset_canonical.len() {
        bitset_canonical[idx]
    } else {
        let (real_idx, mapping) = bitset_canonicalized[idx - bitset_canonical.len()];
        let mut word = bitset_canonical[real_idx as usize];
        let should_invert = mapping & (1 << 6) != 0;
        if should_invert {
            word = !word;
        }
        // Lower 6 bits
        let quantity = mapping & ((1 << 6) - 1);
        if mapping & (1 << 7) != 0 {
            // shift
            word >>= quantity as u64;
        } else {
            word = word.rotate_left(quantity as u32);
        }
        word
    };
    (word & (1 << (needle % 64) as u64)) != 0
}

fn decode_prefix_sum(short_offset_run_header: u32) -> u32 {
    short_offset_run_header & ((1 << 21) - 1)
}

fn decode_length(short_offset_run_header: u32) -> usize {
    (short_offset_run_header >> 21) as usize
}

#[inline(always)]
fn skip_search<const SOR: usize, const OFFSETS: usize>(
    needle: u32,
    short_offset_runs: &[u32; SOR],
    offsets: &[u8; OFFSETS],
) -> bool {
    // Note that this *cannot* be past the end of the array, as the last
    // element is greater than std::char::MAX (the largest possible needle).
    //
    // So, we cannot have found it (i.e. Ok(idx) + 1 != length) and the correct
    // location cannot be past it, so Err(idx) != length either.
    //
    // This means that we can avoid bounds checking for the accesses below, too.
    let last_idx =
        match short_offset_runs.binary_search_by_key(&(needle << 11), |header| header << 11) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };

    let mut offset_idx = decode_length(short_offset_runs[last_idx]);
    let length = if let Some(next) = short_offset_runs.get(last_idx + 1) {
        decode_length(*next) - offset_idx
    } else {
        offsets.len() - offset_idx
    };
    let prev =
        last_idx.checked_sub(1).map(|prev| decode_prefix_sum(short_offset_runs[prev])).unwrap_or(0);

    let total = needle - prev;
    let mut prefix_sum = 0;
    for _ in 0..(length - 1) {
        let offset = offsets[offset_idx];
        prefix_sum += offset as u32;
        if prefix_sum > total {
            break;
        }
        offset_idx += 1;
    }
    offset_idx % 2 == 1
}
