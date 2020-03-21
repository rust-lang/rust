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
    (last_chunk_idx, last_chunk_mapping): (u16, u8),
    bitset_chunk_idx: &[[u8; CHUNK_SIZE]; N1],
    bitset_canonical: &[u64; CANONICAL],
    bitset_canonicalized: &[(u8, u8); CANONICALIZED],
) -> bool {
    let bucket_idx = (needle / 64) as usize;
    let chunk_map_idx = bucket_idx / CHUNK_SIZE;
    let chunk_piece = bucket_idx % CHUNK_SIZE;
    let chunk_idx = if chunk_map_idx >= N {
        if chunk_map_idx == last_chunk_idx as usize {
            last_chunk_mapping
        } else {
            return false;
        }
    } else {
        chunk_idx_map[chunk_map_idx]
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
