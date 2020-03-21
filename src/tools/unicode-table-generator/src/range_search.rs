#[inline(always)]
fn range_search<
    const N: usize,
    const CHUNK_SIZE: usize,
    const N1: usize,
    const CANONICAL: usize,
    const CANONICALIZED: usize,
>(
    needle: u32,
    chunk_idx_map: &[u8; N],
    last_chunk_idx: u16,
    bitset_chunk_idx: &[[u8; CHUNK_SIZE]; N1],
    bitset_canonical: &[u64; CANONICAL],
    bitset_canonicalized: &[(u8, u8); CANONICALIZED],
) -> bool {
    let bucket_idx = (needle / 64) as usize;
    let chunk_map_idx = bucket_idx / CHUNK_SIZE;
    let chunk_piece = bucket_idx % CHUNK_SIZE;
    // The last entry of `chunk_idx_map` actually should be at `last_chunk_idx`,
    // so we need to remap it
    let chunk_idx = if chunk_map_idx < (chunk_idx_map.len() - 1) {
        chunk_idx_map[chunk_map_idx]
    } else if chunk_map_idx == last_chunk_idx as usize {
        chunk_idx_map[chunk_idx_map.len() - 1]
    } else {
        return false;
    };
    let idx = bitset_chunk_idx[(chunk_idx as usize)][chunk_piece] as usize;
    let word = if idx < CANONICAL {
        bitset_canonical[idx]
    } else {
        let (real_idx, mapping) = bitset_canonicalized[idx - CANONICAL];
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
