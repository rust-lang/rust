/// Limits a slice of buffers to at most `n` buffers and ensures that it has at
/// least one buffer, even if empty. When truncating the slice, ensure at least
/// one non-empty buffer remains, to avoid a spurious EOF.
///
/// For example, [POSIX writev] requires that `bufs.len()` is in `1..=IOV_MAX`.
///
/// [POSIX writev]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/writev.html
#[allow(unused_macros)] // Not used on all platforms
pub(crate) macro limit_slices(&mut $bufs:ident, $n:expr) {
    // Rebind $bufs, so the lifetime does not need to live as long as the
    // function parameter, and shadow it so the macro caller gets the result.
    let mut $bufs: &[crate::io::IoSlice<'_>] = $bufs;
    let n: usize = $n;

    let empty = &[crate::io::IoSlice::new(&[])];

    if $bufs.len() > n || $bufs.is_empty() {
        crate::hint::cold_path();
        'fixup: {
            // Find the first non-empty buffer and take up to `n` buffers after it.
            for (i, buf) in $bufs.iter().enumerate() {
                if !buf.is_empty() {
                    let len = crate::cmp::min($bufs.len() - i, n);
                    $bufs = &$bufs[i..i + len];
                    break 'fixup;
                }
            }
            // If no non-empty buffer is found, use a single empty buffer.
            $bufs = empty;
        }
    }
}

/// Limits a slice of buffers to at most `n` buffers and ensures that it has at
/// least one buffer, even if empty. When truncating the slice, ensure at least
/// one non-empty buffer remains, to avoid a spurious EOF.
///
/// For example, [POSIX readv] requires that `bufs.len()` is in `1..=IOV_MAX`.
///
/// [POSIX readv]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/readv.html
#[allow(unused_macros)] // Not used on all platforms
pub(crate) macro limit_slices_mut(&mut $bufs:ident, $n:expr) {
    // Rebind $bufs, so the lifetime does not need to live as long as the
    // function parameter, and shadow it so the macro caller gets the result.
    let mut $bufs: &mut [crate::io::IoSliceMut<'_>] = $bufs;
    let n: usize = $n;

    let empty = &mut [crate::io::IoSliceMut::new(&mut [])];

    if $bufs.len() > n || $bufs.is_empty() {
        crate::hint::cold_path();
        'fixup: {
            // Find the first non-empty buffer and take up to `n` buffers after it.
            for (i, buf) in $bufs.iter().enumerate() {
                if !buf.is_empty() {
                    let len = crate::cmp::min($bufs.len() - i, n);
                    $bufs = &mut $bufs[i..i + len];
                    break 'fixup;
                }
            }
            // If no non-empty buffer is found, use a single empty buffer.
            $bufs = empty;
        }
    }
}
