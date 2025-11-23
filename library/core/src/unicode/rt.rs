//! Runtime support for `unicode_data`.

#[inline(always)]
pub(super) const fn bitset_search<
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
    // FIXME(const-hack): Revert to `slice::get` when slice indexing becomes possible in const.
    let chunk_idx = if chunk_map_idx < chunk_idx_map.len() {
        chunk_idx_map[chunk_map_idx]
    } else {
        return false;
    };
    let idx = bitset_chunk_idx[chunk_idx as usize][chunk_piece] as usize;
    // FIXME(const-hack): Revert to `slice::get` when slice indexing becomes possible in const.
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

#[repr(transparent)]
pub(super) struct ShortOffsetRunHeader(pub(super) u32);

impl ShortOffsetRunHeader {
    pub(super) const fn new(start_index: usize, prefix_sum: u32) -> Self {
        assert!(start_index < (1 << 11));
        assert!(prefix_sum < (1 << 21));

        Self((start_index as u32) << 21 | prefix_sum)
    }

    #[inline]
    pub(super) const fn start_index(&self) -> usize {
        (self.0 >> 21) as usize
    }

    #[inline]
    pub(super) const fn prefix_sum(&self) -> u32 {
        self.0 & ((1 << 21) - 1)
    }
}

/// # Safety
///
/// - The last element of `short_offset_runs` must be greater than `std::char::MAX`.
/// - The start indices of all elements in `short_offset_runs` must be less than `OFFSETS`.
#[inline(always)]
pub(super) unsafe fn skip_search<const SOR: usize, const OFFSETS: usize>(
    needle: char,
    short_offset_runs: &[ShortOffsetRunHeader; SOR],
    offsets: &[u8; OFFSETS],
) -> bool {
    let needle = needle as u32;

    let last_idx =
        match short_offset_runs.binary_search_by_key(&(needle << 11), |header| header.0 << 11) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
    // SAFETY: `last_idx` *cannot* be past the end of the array, as the last
    // element is greater than `std::char::MAX` (the largest possible needle)
    // as guaranteed by the caller.
    //
    // So, we cannot have found it (i.e. `Ok(idx) => idx + 1 != length`) and the
    // correct location cannot be past it, so `Err(idx) => idx != length` either.
    //
    // This means that we can avoid bounds checking for the accesses below, too.
    //
    // We need to use `intrinsics::assume` since the `panic_nounwind` contained
    // in `hint::assert_unchecked` may not be optimized out.
    unsafe { crate::intrinsics::assume(last_idx < SOR) };

    let mut offset_idx = short_offset_runs[last_idx].start_index();
    let length = if let Some(next) = short_offset_runs.get(last_idx + 1) {
        (*next).start_index() - offset_idx
    } else {
        offsets.len() - offset_idx
    };

    let prev =
        last_idx.checked_sub(1).map(|prev| short_offset_runs[prev].prefix_sum()).unwrap_or(0);

    let total = needle - prev;
    let mut prefix_sum = 0;
    for _ in 0..(length - 1) {
        // SAFETY: It is guaranteed that `length <= OFFSETS - offset_idx`,
        // so it follows that `length - 1 + offset_idx < OFFSETS`, therefore
        // `offset_idx < OFFSETS` is always true in this loop.
        //
        // We need to use `intrinsics::assume` since the `panic_nounwind` contained
        // in `hint::assert_unchecked` may not be optimized out.
        unsafe { crate::intrinsics::assume(offset_idx < OFFSETS) };
        let offset = offsets[offset_idx];
        prefix_sum += offset as u32;
        if prefix_sum > total {
            break;
        }
        offset_idx += 1;
    }
    offset_idx % 2 == 1
}

/// # Safety
/// The second component of each tuple in `table` must either be:
/// - A valid `char`
/// - A value with the high bit (1 << 22) set, and the lower 22 bits
///   being a valid index into `multi`.
#[inline(always)]
pub(super) unsafe fn case_conversion(
    c: char,
    ascii_fn: fn(char) -> char,
    table: &[(char, u32)],
    multi: &[[char; 3]],
) -> [char; 3] {
    const INDEX_MASK: u32 = 1 << 22;

    if c.is_ascii() {
        return [ascii_fn(c), '\0', '\0'];
    }

    let Ok(i) = table.binary_search_by(|&(key, _)| key.cmp(&c)) else {
        return [c, '\0', '\0'];
    };

    let u = table[i].1;
    match char::from_u32(u) {
        Option::Some(c) => [c, '\0', '\0'],
        Option::None => {
            // SAFETY: Index comes from statically generated table
            unsafe { *multi.get_unchecked((u & (INDEX_MASK - 1)) as usize) }
        }
    }
}
