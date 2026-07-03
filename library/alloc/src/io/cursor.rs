use crate::alloc::Allocator;
use crate::boxed::Box;
use crate::io::{
    self, Cursor, ErrorKind, IoSlice, WriteThroughCursor, slice_write, slice_write_all,
    slice_write_all_vectored, slice_write_vectored,
};
use crate::vec::Vec;

/// Reserves the required space, and pads the vec with 0s if necessary.
fn reserve_and_pad<A: Allocator>(
    pos_mut: &mut u64,
    vec: &mut Vec<u8, A>,
    buf_len: usize,
) -> io::Result<usize> {
    let pos: usize = (*pos_mut).try_into().map_err(|_| {
        io::const_error!(
            ErrorKind::InvalidInput,
            "cursor position exceeds maximum possible vector length",
        )
    })?;

    // For safety reasons, we don't want these numbers to overflow
    // otherwise our allocation won't be enough
    let desired_cap = pos.saturating_add(buf_len);
    if desired_cap > vec.capacity() {
        // We want our vec's total capacity
        // to have room for (pos+buf_len) bytes. Reserve allocates
        // based on additional elements from the length, so we need to
        // reserve the difference
        cfg_select! {
            no_global_oom_handling => {
                vec.try_reserve(desired_cap - vec.len())?;
            }
            _ => {
                vec.reserve(desired_cap - vec.len());
            }
        }
    }
    // Pad if pos is above the current len.
    if pos > vec.len() {
        let diff = pos - vec.len();
        // Unfortunately, `resize()` would suffice but the optimiser does not
        // realise the `reserve` it does can be eliminated. So we do it manually
        // to eliminate that extra branch
        let spare = vec.spare_capacity_mut();
        debug_assert!(spare.len() >= diff);
        // Safety: we have allocated enough capacity for this.
        // And we are only writing, not reading
        unsafe {
            spare.get_unchecked_mut(..diff).fill(core::mem::MaybeUninit::new(0));
            vec.set_len(pos);
        }
    }

    Ok(pos)
}

/// Writes the slice to the vec without allocating.
///
/// # Safety
///
/// `vec` must have `buf.len()` spare capacity.
unsafe fn vec_write_all_unchecked<A>(pos: usize, vec: &mut Vec<u8, A>, buf: &[u8]) -> usize
where
    A: Allocator,
{
    debug_assert!(vec.capacity() >= pos + buf.len());
    unsafe { vec.as_mut_ptr().add(pos).copy_from(buf.as_ptr(), buf.len()) };
    pos + buf.len()
}

/// Resizing `write_all` implementation for [`Cursor`].
///
/// Cursor is allowed to have a pre-allocated and initialised
/// vector body, but with a position of 0. This means the [`Write`]
/// will overwrite the contents of the vec.
///
/// This also allows for the vec body to be empty, but with a position of N.
/// This means that [`Write`] will pad the vec with 0 initially,
/// before writing anything from that point
///
/// [`Write`]: crate::io::Write
fn vec_write_all<A>(pos_mut: &mut u64, vec: &mut Vec<u8, A>, buf: &[u8]) -> io::Result<usize>
where
    A: Allocator,
{
    let buf_len = buf.len();
    let mut pos = reserve_and_pad(pos_mut, vec, buf_len)?;

    // Write the buf then progress the vec forward if necessary
    // Safety: we have ensured that the capacity is available
    // and that all bytes get written up to pos
    unsafe {
        pos = vec_write_all_unchecked(pos, vec, buf);
        if pos > vec.len() {
            vec.set_len(pos);
        }
    };

    // Bump us forward
    *pos_mut += buf_len as u64;
    Ok(buf_len)
}

/// Resizing `write_all_vectored` implementation for [`Cursor`].
///
/// Cursor is allowed to have a pre-allocated and initialised
/// vector body, but with a position of 0. This means the [`Write`]
/// will overwrite the contents of the vec.
///
/// This also allows for the vec body to be empty, but with a position of N.
/// This means that [`Write`] will pad the vec with 0 initially,
/// before writing anything from that point
///
/// [`Write`]: crate::io::Write
fn vec_write_all_vectored<A>(
    pos_mut: &mut u64,
    vec: &mut Vec<u8, A>,
    bufs: &[IoSlice<'_>],
) -> io::Result<usize>
where
    A: Allocator,
{
    // For safety reasons, we don't want this sum to overflow ever.
    // If this saturates, the reserve should panic to avoid any unsound writing.
    let buf_len = bufs.iter().fold(0usize, |a, b| a.saturating_add(b.len()));
    let mut pos = reserve_and_pad(pos_mut, vec, buf_len)?;

    // Write the buf then progress the vec forward if necessary
    // Safety: we have ensured that the capacity is available
    // and that all bytes get written up to the last pos
    unsafe {
        for buf in bufs {
            pos = vec_write_all_unchecked(pos, vec, buf);
        }
        if pos > vec.len() {
            vec.set_len(pos);
        }
    }

    // Bump us forward
    *pos_mut += buf_len as u64;
    Ok(buf_len)
}

#[stable(feature = "cursor_mut_vec", since = "1.25.0")]
impl<A> WriteThroughCursor for &mut Vec<u8, A>
where
    A: Allocator,
{
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)
    }

    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(_this: &Cursor<Self>) -> bool {
        true
    }

    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)?;
        Ok(())
    }

    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)?;
        Ok(())
    }

    #[inline]
    fn flush(_this: &mut Cursor<Self>) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> WriteThroughCursor for Vec<u8, A>
where
    A: Allocator,
{
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)
    }

    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(_this: &Cursor<Self>) -> bool {
        true
    }

    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)?;
        Ok(())
    }

    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)?;
        Ok(())
    }

    #[inline]
    fn flush(_this: &mut Cursor<Self>) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "cursor_box_slice", since = "1.5.0")]
impl<A> WriteThroughCursor for Box<[u8], A>
where
    A: Allocator,
{
    #[inline]
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        slice_write(pos, inner, buf)
    }

    #[inline]
    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        slice_write_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(_this: &Cursor<Self>) -> bool {
        true
    }

    #[inline]
    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        slice_write_all(pos, inner, buf)
    }

    #[inline]
    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        slice_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn flush(_this: &mut Cursor<Self>) -> io::Result<()> {
        Ok(())
    }
}
