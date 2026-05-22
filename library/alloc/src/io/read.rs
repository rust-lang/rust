use core::cmp;
use core::mem::{DropGuard, MaybeUninit};

use crate::io::{
    BorrowedBuf, BorrowedCursor, Bytes, Chain, Error, IoSliceMut, Result, Take, bytes, chain, take,
};
use crate::string::String;
use crate::vec::Vec;

/// The `Read` trait allows for reading bytes from a source.
///
/// Implementors of the `Read` trait are called 'readers'.
///
/// Readers are defined by one required method, [`read()`]. Each call to [`read()`]
/// will attempt to pull bytes from this source into a provided buffer. A
/// number of other methods are implemented in terms of [`read()`], giving
/// implementors a number of ways to read bytes while only needing to implement
/// a single method.
///
/// Readers are intended to be composable with one another. Many implementors
/// throughout [`std::io`] take and provide types which implement the `Read`
/// trait.
///
/// Please note that each call to [`read()`] may involve a system call, and
/// `BufReader`, will be more efficient.
/// therefore, using something that implements [`BufRead`], such as
///
/// [`BufRead`]: crate::io::BufRead
///
/// Repeated calls to the reader use the same cursor, so for example
/// calling `read_to_end` twice on a `File` will only return the file's
/// contents once. It's recommended to first call `rewind()` in that case.
///
/// # Examples
///
/// `File`s implement `Read`:
///
/// ```no_run
/// use std::io;
/// use std::io::prelude::*;
/// use std::fs::File;
///
/// fn main() -> io::Result<()> {
///     let mut f = File::open("foo.txt")?;
///     let mut buffer = [0; 10];
///
///     // read up to 10 bytes
///     f.read(&mut buffer)?;
///
///     let mut buffer = Vec::new();
///     // read the whole file
///     f.read_to_end(&mut buffer)?;
///
///     // read into a String, so that you don't need to do the conversion.
///     let mut buffer = String::new();
///     f.read_to_string(&mut buffer)?;
///
///     // and more! See the other methods for more details.
///     Ok(())
/// }
/// ```
///
/// Read from [`&str`] because [`&[u8]`][prim@slice] implements `Read`:
///
/// ```no_run
/// # use std::io;
/// use std::io::prelude::*;
///
/// fn main() -> io::Result<()> {
///     let mut b = "This string will be read".as_bytes();
///     let mut buffer = [0; 10];
///
///     // read up to 10 bytes
///     b.read(&mut buffer)?;
///
///     // etc... it works exactly as a File does!
///     Ok(())
/// }
/// ```
///
/// [`read()`]: Read::read
/// [`&str`]: prim@str
/// [`std::io`]: crate::io
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(notable_trait)]
#[cfg_attr(not(test), rustc_diagnostic_item = "IoRead")]
pub trait Read {
    /// Pull some bytes from this source into the specified buffer, returning
    /// how many bytes were read.
    ///
    /// This function does not provide any guarantees about whether it blocks
    /// waiting for data, but if an object needs to block for a read and cannot,
    /// it will typically signal this via an [`Err`] return value.
    ///
    /// If the return value of this method is [`Ok(n)`], then implementations must
    /// guarantee that `0 <= n <= buf.len()`. A nonzero `n` value indicates
    /// that the buffer `buf` has been filled in with `n` bytes of data from this
    /// source. If `n` is `0`, then it can indicate one of two scenarios:
    ///
    /// 1. This reader has reached its "end of file" and will likely no longer
    ///    be able to produce bytes. Note that this does not mean that the
    ///    reader will *always* no longer be able to produce bytes. As an example,
    ///    on Linux, this method will call the `recv` syscall for a `TcpStream`,
    ///    where returning zero indicates the connection was shut down correctly. While
    ///    for `File`, it is possible to reach the end of file and get zero as result,
    ///    but if more data is appended to the file, future calls to `read` will return
    ///    more data.
    /// 2. The buffer specified was 0 bytes in length.
    ///
    /// It is not an error if the returned value `n` is smaller than the buffer size,
    /// even when the reader is not at the end of the stream yet.
    /// This may happen for example because fewer bytes are actually available right now
    /// (e. g. being close to end-of-file) or because read() was interrupted by a signal.
    ///
    /// As this trait is safe to implement, callers in unsafe code cannot rely on
    /// `n <= buf.len()` for safety.
    /// Extra care needs to be taken when `unsafe` functions are used to access the read bytes.
    /// Callers have to ensure that no unchecked out-of-bounds accesses are possible even if
    /// `n > buf.len()`.
    ///
    /// *Implementations* of this method can make no assumptions about the contents of `buf` when
    /// this function is called. It is recommended that implementations only write data to `buf`
    /// instead of reading its contents.
    ///
    /// Correspondingly, however, *callers* of this method in unsafe code must not assume
    /// any guarantees about how the implementation uses `buf`. The trait is safe to implement,
    /// so it is possible that the code that's supposed to write to the buffer might also read
    /// from it. It is your responsibility to make sure that `buf` is initialized
    /// before calling `read`. Calling `read` with an uninitialized `buf` (of the kind one
    /// obtains via [`MaybeUninit<T>`]) is not safe, and can lead to undefined behavior.
    ///
    /// [`MaybeUninit<T>`]: core::mem::MaybeUninit
    ///
    /// # Errors
    ///
    /// If this function encounters any form of I/O or other error, an error
    /// variant will be returned. If an error is returned then it must be
    /// guaranteed that no bytes were read.
    ///
    /// An error of the [`ErrorKind::Interrupted`] kind is non-fatal and the read
    /// operation should be retried if there is nothing else to do.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// [`Ok(n)`]: Ok
    /// [`ErrorKind::Interrupted`]: crate::io::ErrorKind::Interrupted
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let mut buffer = [0; 10];
    ///
    ///     // read up to 10 bytes
    ///     let n = f.read(&mut buffer[..])?;
    ///
    ///     println!("The bytes: {:?}", &buffer[..n]);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    /// Like `read`, except that it reads into a slice of buffers.
    ///
    /// Data is copied to fill each buffer in order, with the final buffer
    /// written to possibly being only partially filled. This method must
    /// behave equivalently to a single call to `read` with concatenated
    /// buffers.
    ///
    /// The default implementation calls `read` with either the first nonempty
    /// buffer provided, or an empty one if none exists.
    #[stable(feature = "iovec", since = "1.36.0")]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> Result<usize> {
        default_read_vectored(|b| self.read(b), bufs)
    }

    /// Determines if this `Read`er has an efficient `read_vectored`
    /// implementation.
    ///
    /// If a `Read`er does not override the default `read_vectored`
    /// implementation, code using it may want to avoid the method all together
    /// and coalesce writes into a single buffer for higher performance.
    ///
    /// The default implementation returns `false`.
    #[unstable(feature = "can_vector", issue = "69941")]
    fn is_read_vectored(&self) -> bool {
        false
    }

    /// Reads all bytes until EOF in this source, placing them into `buf`.
    ///
    /// All bytes read from this source will be appended to the specified buffer
    /// `buf`. This function will continuously call [`read()`] to append more data to
    /// `buf` until [`read()`] returns either [`Ok(0)`] or an error of
    /// non-[`ErrorKind::Interrupted`] kind.
    ///
    /// If successful, this function will return the total number of bytes read.
    ///
    /// # Errors
    ///
    /// If this function encounters an error of the kind
    /// [`ErrorKind::Interrupted`] then the error is ignored and the operation
    /// will continue.
    ///
    /// If any other read error is encountered then this function immediately
    /// returns. Any bytes which have already been read will be appended to
    /// `buf`.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// [`Ok(0)`]: Ok
    /// [`ErrorKind::Interrupted`]: crate::io::ErrorKind::Interrupted
    /// [`read()`]: Read::read
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let mut buffer = Vec::new();
    ///
    ///     // read the whole file
    ///     f.read_to_end(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// (See also the `std::fs::read` convenience function for reading from a
    /// file.)
    ///
    /// ## Implementing `read_to_end`
    ///
    /// When implementing the `io::Read` trait, it is recommended to allocate
    /// memory using [`Vec::try_reserve`]. However, this behavior is not guaranteed
    /// by all implementations, and `read_to_end` may not handle out-of-memory
    /// situations gracefully.
    ///
    /// ```no_run
    /// # #![expect(dead_code)]
    /// # use std::io::{self, BufRead};
    /// # struct Example { example_datasource: io::Empty } impl Example {
    /// # fn get_some_data_for_the_example(&self) -> &'static [u8] { &[] }
    /// fn read_to_end(&mut self, dest_vec: &mut Vec<u8>) -> io::Result<usize> {
    ///     let initial_vec_len = dest_vec.len();
    ///     loop {
    ///         let src_buf = self.example_datasource.fill_buf()?;
    ///         if src_buf.is_empty() {
    ///             break;
    ///         }
    ///         dest_vec.try_reserve(src_buf.len())?;
    ///         dest_vec.extend_from_slice(src_buf);
    ///
    ///         // Any irreversible side effects should happen after `try_reserve` succeeds,
    ///         // to avoid losing data on allocation error.
    ///         let read = src_buf.len();
    ///         self.example_datasource.consume(read);
    ///     }
    ///     Ok(dest_vec.len() - initial_vec_len)
    /// }
    /// # }
    /// ```
    ///
    /// # Usage Notes
    ///
    /// `read_to_end` attempts to read a source until EOF, but many sources are continuous streams
    /// that do not send EOF. In these cases, `read_to_end` will block indefinitely. Standard input
    /// is one such stream which may be finite if piped, but is typically continuous. For example,
    /// `cat file | my-rust-program` will correctly terminate with an `EOF` upon closure of cat.
    /// Reading user input or running programs that remain open indefinitely will never terminate
    /// the stream with `EOF` (e.g. `yes | my-rust-program`).
    ///
    /// Using `.lines()` with a `BufReader` or using [`read`] can provide a better solution
    ///
    /// [`read`]: Read::read
    /// [`Vec::try_reserve`]: crate::vec::Vec::try_reserve
    #[stable(feature = "rust1", since = "1.0.0")]
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize> {
        default_read_to_end(self, buf, None)
    }

    /// Reads all bytes until EOF in this source, appending them to `buf`.
    ///
    /// If successful, this function returns the number of bytes which were read
    /// and appended to `buf`.
    ///
    /// # Errors
    ///
    /// If the data in this stream is *not* valid UTF-8 then an error is
    /// returned and `buf` is unchanged.
    ///
    /// See [`read_to_end`] for other error semantics.
    ///
    /// [`read_to_end`]: Read::read_to_end
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let mut buffer = String::new();
    ///
    ///     f.read_to_string(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// (See also the `std::fs::read_to_string` convenience function for
    /// reading from a file.)
    ///
    /// # Usage Notes
    ///
    /// `read_to_string` attempts to read a source until EOF, but many sources are continuous streams
    /// that do not send EOF. In these cases, `read_to_string` will block indefinitely. Standard input
    /// is one such stream which may be finite if piped, but is typically continuous. For example,
    /// `cat file | my-rust-program` will correctly terminate with an `EOF` upon closure of cat.
    /// Reading user input or running programs that remain open indefinitely will never terminate
    /// the stream with `EOF` (e.g. `yes | my-rust-program`).
    ///
    /// Using `.lines()` with a `BufReader` or using [`read`] can provide a better solution
    ///
    /// [`read`]: Read::read
    #[stable(feature = "rust1", since = "1.0.0")]
    fn read_to_string(&mut self, buf: &mut String) -> Result<usize> {
        default_read_to_string(self, buf, None)
    }

    /// Reads the exact number of bytes required to fill `buf`.
    ///
    /// This function reads as many bytes as necessary to completely fill the
    /// specified buffer `buf`.
    ///
    /// *Implementations* of this method can make no assumptions about the contents of `buf` when
    /// this function is called. It is recommended that implementations only write data to `buf`
    /// instead of reading its contents. The documentation on [`read`] has a more detailed
    /// explanation of this subject.
    ///
    /// # Errors
    ///
    /// If this function encounters an error of the kind
    /// [`ErrorKind::Interrupted`] then the error is ignored and the operation
    /// will continue.
    ///
    /// If this function encounters an "end of file" before completely filling
    /// the buffer, it returns an error of the kind [`ErrorKind::UnexpectedEof`].
    /// The contents of `buf` are unspecified in this case.
    ///
    /// If any other read error is encountered then this function immediately
    /// returns. The contents of `buf` are unspecified in this case.
    ///
    /// If this function returns an error, it is unspecified how many bytes it
    /// has read, but it will never read more than would be necessary to
    /// completely fill the buffer.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// [`ErrorKind::Interrupted`]: crate::io::ErrorKind::Interrupted
    /// [`ErrorKind::UnexpectedEof`]: crate::io::ErrorKind::UnexpectedEof
    /// [`read`]: Read::read
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let mut buffer = [0; 10];
    ///
    ///     // read exactly 10 bytes
    ///     f.read_exact(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "read_exact", since = "1.6.0")]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        default_read_exact(self, buf)
    }

    /// Pull some bytes from this source into the specified buffer.
    ///
    /// This is equivalent to the [`read`](Read::read) method, except that it is passed a [`BorrowedCursor`] rather than `[u8]` to allow use
    /// with uninitialized buffers. The new data will be appended to any existing contents of `buf`.
    ///
    /// The default implementation delegates to `read`.
    ///
    /// This method makes it possible to return both data and an error but it is advised against.
    #[unstable(feature = "read_buf", issue = "78485")]
    fn read_buf(&mut self, buf: BorrowedCursor<'_, u8>) -> Result<()> {
        default_read_buf(|b| self.read(b), buf)
    }

    /// Reads the exact number of bytes required to fill `cursor`.
    ///
    /// This is similar to the [`read_exact`](Read::read_exact) method, except
    /// that it is passed a [`BorrowedCursor`] rather than `[u8]` to allow use
    /// with uninitialized buffers.
    ///
    /// # Errors
    ///
    /// If this function encounters an error of the kind [`ErrorKind::Interrupted`]
    /// then the error is ignored and the operation will continue.
    ///
    /// If this function encounters an "end of file" before completely filling
    /// the buffer, it returns an error of the kind [`ErrorKind::UnexpectedEof`].
    ///
    /// If any other read error is encountered then this function immediately
    /// returns.
    ///
    /// If this function returns an error, all bytes read will be appended to `cursor`.
    ///
    /// [`ErrorKind::Interrupted`]: crate::io::ErrorKind::Interrupted
    /// [`ErrorKind::UnexpectedEof`]: crate::io::ErrorKind::UnexpectedEof
    #[unstable(feature = "read_buf", issue = "78485")]
    fn read_buf_exact(&mut self, cursor: BorrowedCursor<'_, u8>) -> Result<()> {
        default_read_buf_exact(self, cursor)
    }

    /// Creates a "by reference" adapter for this instance of `Read`.
    ///
    /// The returned adapter also implements `Read` and will simply borrow this
    /// current reader.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::Read;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::open("foo.txt")?;
    ///     let mut buffer = Vec::new();
    ///     let mut other_buffer = Vec::new();
    ///
    ///     {
    ///         let reference = f.by_ref();
    ///
    ///         // read at most 5 bytes
    ///         reference.take(5).read_to_end(&mut buffer)?;
    ///
    ///     } // drop our &mut reference so we can use f again
    ///
    ///     // original file still usable, read the rest
    ///     f.read_to_end(&mut other_buffer)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn by_ref(&mut self) -> &mut Self
    where
        Self: Sized,
    {
        self
    }

    /// Transforms this `Read` instance to an [`Iterator`] over its bytes.
    ///
    /// The returned type implements [`Iterator`] where the [`Item`] is
    /// <code>[Result]<[u8], [io::Error]></code>.
    /// The yielded item is [`Ok`] if a byte was successfully read and [`Err`]
    /// otherwise. EOF is mapped to returning [`None`] from this iterator.
    ///
    /// The default implementation calls `read` for each byte,
    /// which can be very inefficient for data that's not in memory,
    /// such as `File`. Consider using a `BufReader` in such cases.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// [`Item`]: Iterator::Item
    /// [Result]: core::result::Result "Result"
    /// [io::Error]: crate::io::Error "io::Error"
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let f = BufReader::new(File::open("foo.txt")?);
    ///
    ///     for byte in f.bytes() {
    ///         println!("{}", byte?);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bytes(self) -> Bytes<Self>
    where
        Self: Sized,
    {
        bytes(self)
    }

    /// Creates an adapter which will chain this stream with another.
    ///
    /// The returned `Read` instance will first read all bytes from this object
    /// until EOF is encountered. Afterwards the output is equivalent to the
    /// output of `next`.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let f1 = File::open("foo.txt")?;
    ///     let f2 = File::open("bar.txt")?;
    ///
    ///     let mut handle = f1.chain(f2);
    ///     let mut buffer = String::new();
    ///
    ///     // read the value into a String. We could use any Read method here,
    ///     // this is just one example.
    ///     handle.read_to_string(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chain<R: Read>(self, next: R) -> Chain<Self, R>
    where
        Self: Sized,
    {
        chain(self, next)
    }

    /// Creates an adapter which will read at most `limit` bytes from it.
    ///
    /// This function returns a new instance of `Read` which will read at most
    /// `limit` bytes, after which it will always return EOF ([`Ok(0)`]). Any
    /// read errors will not count towards the number of bytes read and future
    /// calls to [`read()`] may succeed.
    ///
    /// # Examples
    ///
    /// `File`s implement `Read`:
    ///
    /// [`Ok(0)`]: Ok
    /// [`read()`]: Read::read
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    /// use std::fs::File;
    ///
    /// fn main() -> io::Result<()> {
    ///     let f = File::open("foo.txt")?;
    ///     let mut buffer = [0; 5];
    ///
    ///     // read at most five bytes
    ///     let mut handle = f.take(5);
    ///
    ///     handle.read(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take(self, limit: u64) -> Take<Self>
    where
        Self: Sized,
    {
        take(self, limit)
    }

    /// Read and return a fixed array of bytes from this source.
    ///
    /// This function uses an array sized based on a const generic size known at compile time. You
    /// can specify the size with turbofish (`reader.read_array::<8>()`), or let type inference
    /// determine the number of bytes needed based on how the return value gets used. For instance,
    /// this function works well with functions like [`u64::from_le_bytes`] to turn an array of
    /// bytes into an integer of the same size.
    ///
    /// Like `read_exact`, if this function encounters an "end of file" before reading the desired
    /// number of bytes, it returns an error of the kind [`ErrorKind::UnexpectedEof`].
    ///
    /// [`ErrorKind::UnexpectedEof`]: crate::io::ErrorKind::UnexpectedEof
    ///
    /// ```
    /// #![feature(read_array)]
    /// use std::io::Cursor;
    /// use std::io::prelude::*;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut buf = Cursor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2]);
    ///     let x = u64::from_le_bytes(buf.read_array()?);
    ///     let y = u32::from_be_bytes(buf.read_array()?);
    ///     let z = u16::from_be_bytes(buf.read_array()?);
    ///     assert_eq!(x, 0x807060504030201);
    ///     assert_eq!(y, 0x9080706);
    ///     assert_eq!(z, 0x504);
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "read_array", issue = "148848")]
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]>
    where
        Self: Sized,
    {
        let mut buf = [MaybeUninit::uninit(); N];
        let mut borrowed_buf = BorrowedBuf::from(buf.as_mut_slice());
        self.read_buf_exact(borrowed_buf.unfilled())?;
        // Guard against incorrect `read_buf_exact` implementations.
        assert_eq!(borrowed_buf.len(), N);
        Ok(unsafe { MaybeUninit::array_assume_init(buf) })
    }

    /// Read and return a type (e.g. an integer) in little-endian order.
    ///
    /// You can specify the type with turbofish (`reader.read_le::<u64>()`), or let type inference
    /// determine the type based on how the return value gets used.
    ///
    /// Like `read_exact`, if this function encounters an "end of file" before reading the desired
    /// number of bytes, it returns an error of the kind [`ErrorKind::UnexpectedEof`].
    ///
    /// [`ErrorKind::UnexpectedEof`]: crate::io::ErrorKind::UnexpectedEof
    ///
    /// ```
    /// #![feature(read_le)]
    /// use std::io::Cursor;
    /// use std::io::prelude::*;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut buf = Cursor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2]);
    ///     let x: u64 = buf.read_le()?;
    ///     let y: u32 = buf.read_le()?;
    ///     let z = buf.read_le::<u16>()?;
    ///     assert_eq!(x, 0x807060504030201);
    ///     assert_eq!(y, 0x6070809);
    ///     assert_eq!(z, 0x405);
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "read_le", issue = "156984")]
    #[inline]
    fn read_le<T: FromEndianBytes>(&mut self) -> Result<T>
    where
        Self: Sized,
    {
        T::read_le_from(self)
    }

    /// Read and return a type (e.g. an integer) in big-endian order.
    ///
    /// You can specify the type with turbofish (`reader.read_be::<u64>()`), or let type inference
    /// determine the type based on how the return value gets used.
    ///
    /// Like `read_exact`, if this function encounters an "end of file" before reading the desired
    /// number of bytes, it returns an error of the kind [`ErrorKind::UnexpectedEof`].
    ///
    /// [`ErrorKind::UnexpectedEof`]: crate::io::ErrorKind::UnexpectedEof
    ///
    /// ```
    /// #![feature(read_le)]
    /// use std::io::Cursor;
    /// use std::io::prelude::*;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut buf = Cursor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2]);
    ///     let x: u64 = buf.read_be()?;
    ///     let y: u32 = buf.read_be()?;
    ///     let z = buf.read_be::<u16>()?;
    ///     assert_eq!(x, 0x102030405060708);
    ///     assert_eq!(y, 0x9080706);
    ///     assert_eq!(z, 0x504);
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "read_le", issue = "156984")]
    #[inline]
    fn read_be<T: FromEndianBytes>(&mut self) -> Result<T>
    where
        Self: Sized,
    {
        T::read_be_from(self)
    }
}

/// Reads all bytes from a [reader][Read] into a new [`String`].
///
/// This is a convenience function for [`Read::read_to_string`]. Using this
/// function avoids having to create a variable first and provides more type
/// safety since you can only get the buffer out if there were no errors. (If you
/// use [`Read::read_to_string`] you have to remember to check whether the read
/// succeeded because otherwise your buffer will be empty or only partially full.)
///
/// # Performance
///
/// The downside of this function's increased ease of use and type safety is
/// that it gives you less control over performance. For example, you can't
/// pre-allocate memory like you can using [`String::with_capacity`] and
/// [`Read::read_to_string`]. Also, you can't re-use the buffer if an error
/// occurs while reading.
///
/// In many cases, this function's performance will be adequate and the ease of use
/// and type safety tradeoffs will be worth it. However, there are cases where you
/// need more control over performance, and in those cases you should definitely use
/// [`Read::read_to_string`] directly.
///
/// Note that in some special cases, such as when reading files, this function will
/// pre-allocate memory based on the size of the input it is reading. In those
/// cases, the performance should be as good as if you had used
/// [`Read::read_to_string`] with a manually pre-allocated buffer.
///
/// # Errors
///
/// This function forces you to handle errors because the output (the `String`)
/// is wrapped in a [`Result`]. See [`Read::read_to_string`] for the errors
/// that can occur. If any error occurs, you will get an [`Err`], so you
/// don't have to worry about your buffer being empty or partially full.
///
/// # Examples
///
/// ```no_run
/// # use std::io;
/// fn main() -> io::Result<()> {
///     let stdin = io::read_to_string(io::stdin())?;
///     println!("Stdin was:");
///     println!("{stdin}");
///     Ok(())
/// }
/// ```
///
/// # Usage Notes
///
/// `read_to_string` attempts to read a source until EOF, but many sources are continuous streams
/// that do not send EOF. In these cases, `read_to_string` will block indefinitely. Standard input
/// is one such stream which may be finite if piped, but is typically continuous. For example,
/// `cat file | my-rust-program` will correctly terminate with an `EOF` upon closure of cat.
/// Reading user input or running programs that remain open indefinitely will never terminate
/// the stream with `EOF` (e.g. `yes | my-rust-program`).
///
/// Using `.lines()` with a `BufReader` or using [`read`] can provide a better solution
///
/// [`read`]: Read::read
///
#[stable(feature = "io_read_to_string", since = "1.65.0")]
pub fn read_to_string<R: Read>(mut reader: R) -> Result<String> {
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    Ok(buf)
}

/// Bare metal platforms usually have very small amounts of RAM
/// (in the order of hundreds of KB)
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub const DEFAULT_BUF_SIZE: usize = cfg_select! {
    target_os = "espidf" => { 512 },
    _ => { 8 * 1024 }
};

/// Several `read_to_string` and `read_line` methods in the standard library will
/// append data into a `String` buffer, but we need to be pretty careful when
/// doing this. The implementation will just call `.as_mut_vec()` and then
/// delegate to a byte-oriented reading method, but we must ensure that when
/// returning we never leave `buf` in a state such that it contains invalid UTF-8
/// in its bounds.
///
/// To this end, we use an RAII guard (to protect against panics) which updates
/// the length of the string when it is dropped. This guard initially truncates
/// the string to the prior length and only after we've validated that the
/// new contents are valid UTF-8 do we allow it to set a longer length.
///
/// The unsafety in this function is twofold:
///
/// 1. We're looking at the raw bytes of `buf`, so we take on the burden of UTF-8
///    checks.
/// 2. We're passing a raw buffer to the function `f`, and it is expected that
///    the function only *appends* bytes to the buffer. We'll get undefined
///    behavior if existing bytes are overwritten to have non-UTF-8 data.
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub unsafe fn append_to_string<F>(buf: &mut String, f: F) -> Result<usize>
where
    F: FnOnce(&mut Vec<u8>) -> Result<usize>,
{
    let len_original = buf.len();
    // SAFETY: invalid UTF-8 discarded before return or unwind
    let buf_vec = unsafe { buf.as_mut_vec() };
    let mut g = DropGuard::new((len_original, buf_vec), |(len, buf)| unsafe {
        buf.set_len(len);
    });
    let ret = f(g.1);

    // SAFETY: the caller promises to only append data to `buf`
    let appended = unsafe { g.1.get_unchecked(g.0..) };
    if str::from_utf8(appended).is_err() {
        ret.and_then(|_| Err(Error::INVALID_UTF8))
    } else {
        g.0 = g.1.len();
        ret
    }
}

/// Here we must serve many masters with conflicting goals:
///
/// - avoid allocating unless necessary
/// - avoid overallocating if we know the exact size (#89165)
/// - avoid passing large buffers to readers that always initialize the free capacity if they perform short reads (#23815, #23820)
/// - pass large buffers to readers that do not initialize the spare capacity. this can amortize per-call overheads
/// - and finally pass not-too-small and not-too-large buffers to Windows read APIs because they manage to suffer from both problems
///   at the same time, i.e. small reads suffer from syscall overhead, all reads incur costs proportional to buffer size (#110650)
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_read_to_end<R: Read + ?Sized>(
    r: &mut R,
    buf: &mut Vec<u8>,
    size_hint: Option<usize>,
) -> Result<usize> {
    let start_len = buf.len();
    let start_cap = buf.capacity();
    // Optionally limit the maximum bytes read on each iteration.
    // This adds an arbitrary fiddle factor to allow for more data than we expect.
    let mut max_read_size = size_hint
        .and_then(|s| s.checked_add(1024)?.checked_next_multiple_of(DEFAULT_BUF_SIZE))
        .unwrap_or(DEFAULT_BUF_SIZE);

    const PROBE_SIZE: usize = 32;

    fn small_probe_read<R: Read + ?Sized>(r: &mut R, buf: &mut Vec<u8>) -> Result<usize> {
        let mut probe = [0u8; PROBE_SIZE];

        loop {
            cfg_select! {
                no_global_oom_handling => {
                    // Without global OOM handling we must proactively allocate the buffer
                    // to avoid failing after already reading data.
                    buf.try_reserve(PROBE_SIZE)?;
                }
                _ => {}
            }

            match r.read(&mut probe) {
                Ok(n) => {
                    cfg_select! {
                        no_global_oom_handling => {
                            // there is no way to recover from allocation failure here
                            // because the data has already been read.
                            buf.try_extend_from_slice_of_bytes(&probe[..n])?;
                        }
                        _ => {
                            // there is no way to recover from allocation failure here
                            // because the data has already been read.
                            buf.extend_from_slice(&probe[..n]);
                        }
                    }
                    return Ok(n);
                }
                Err(ref e) if e.is_interrupted() => continue,
                Err(e) => return Err(e),
            }
        }
    }

    // avoid inflating empty/small vecs before we have determined that there's anything to read
    if (size_hint.is_none() || size_hint == Some(0)) && buf.capacity() - buf.len() < PROBE_SIZE {
        let read = small_probe_read(r, buf)?;

        if read == 0 {
            return Ok(0);
        }
    }

    loop {
        if buf.len() == buf.capacity() && buf.capacity() == start_cap {
            // The buffer might be an exact fit. Let's read into a probe buffer
            // and see if it returns `Ok(0)`. If so, we've avoided an
            // unnecessary doubling of the capacity. But if not, append the
            // probe buffer to the primary buffer and let its capacity grow.
            let read = small_probe_read(r, buf)?;

            if read == 0 {
                return Ok(buf.len() - start_len);
            }
        }

        if buf.len() == buf.capacity() {
            // buf is full, need more space
            buf.try_reserve(PROBE_SIZE)?;
        }

        let mut spare = buf.spare_capacity_mut();
        let buf_len = cmp::min(spare.len(), max_read_size);
        spare = &mut spare[..buf_len];
        let mut read_buf: BorrowedBuf<'_, u8> = spare.into();

        // Note that we don't track already initialized bytes here, but this is fine
        // because we explicitly limit the read size
        let mut cursor = read_buf.unfilled();
        let result = loop {
            match r.read_buf(cursor.reborrow()) {
                Err(e) if e.is_interrupted() => continue,
                // Do not stop now in case of error: we might have received both data
                // and an error
                res => break res,
            }
        };

        let bytes_read = cursor.written();
        let is_init = read_buf.is_init();

        // SAFETY: BorrowedBuf's invariants mean this much memory is initialized.
        unsafe {
            let new_len = bytes_read + buf.len();
            buf.set_len(new_len);
        }

        // Now that all data is pushed to the vector, we can fail without data loss
        result?;

        if bytes_read == 0 {
            return Ok(buf.len() - start_len);
        }

        // Use heuristics to determine the max read size if no initial size hint was provided
        if size_hint.is_none() {
            // The reader is returning short reads but it doesn't call ensure_init().
            // In that case we no longer need to restrict read sizes to avoid
            // initialization costs.
            // When reading from disk we usually don't get any short reads except at EOF.
            // So we wait for at least 2 short reads before uncapping the read buffer;
            // this helps with the Windows issue.
            if !is_init {
                max_read_size = usize::MAX;
            }
            // we have passed a larger buffer than previously and the
            // reader still hasn't returned a short read
            else if buf_len >= max_read_size && bytes_read == buf_len {
                max_read_size = max_read_size.saturating_mul(2);
            }
        }
    }
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_read_to_string<R: Read + ?Sized>(
    r: &mut R,
    buf: &mut String,
    size_hint: Option<usize>,
) -> Result<usize> {
    // Note that we do *not* call `r.read_to_end()` here. We are passing
    // `&mut Vec<u8>` (the raw contents of `buf`) into the `read_to_end`
    // method to fill it up. An arbitrary implementation could overwrite the
    // entire contents of the vector, not just append to it (which is what
    // we are expecting).
    //
    // To prevent extraneously checking the UTF-8-ness of the entire buffer
    // we pass it to our hardcoded `default_read_to_end` implementation which
    // we know is guaranteed to only read data into the end of the buffer.
    unsafe { append_to_string(buf, |b| default_read_to_end(r, b, size_hint)) }
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_read_vectored<F>(read: F, bufs: &mut [IoSliceMut<'_>]) -> Result<usize>
where
    F: FnOnce(&mut [u8]) -> Result<usize>,
{
    let buf = bufs.iter_mut().find(|b| !b.is_empty()).map_or(&mut [][..], |b| &mut **b);
    read(buf)
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_read_exact<R: Read + ?Sized>(this: &mut R, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        match this.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                buf = &mut buf[n..];
            }
            Err(ref e) if e.is_interrupted() => {}
            Err(e) => return Err(e),
        }
    }
    if !buf.is_empty() { Err(Error::READ_EXACT_EOF) } else { Ok(()) }
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_read_buf<F>(read: F, mut cursor: BorrowedCursor<'_, u8>) -> Result<()>
where
    F: FnOnce(&mut [u8]) -> Result<usize>,
{
    let n = read(cursor.ensure_init())?;
    cursor.advance_checked(n);
    Ok(())
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_read_buf_exact<R: Read + ?Sized>(
    this: &mut R,
    mut cursor: BorrowedCursor<'_, u8>,
) -> Result<()> {
    while cursor.capacity() > 0 {
        let prev_written = cursor.written();
        match this.read_buf(cursor.reborrow()) {
            Ok(()) => {}
            Err(e) if e.is_interrupted() => continue,
            Err(e) => return Err(e),
        }

        if cursor.written() == prev_written {
            return Err(Error::READ_EXACT_EOF);
        }
    }

    Ok(())
}

mod sealed {
    /// This trait being unreachable from outside the crate
    /// prevents outside implementations of our extension traits.
    /// This allows adding more trait methods in the future.
    #[unstable(feature = "sealed", issue = "none")]
    pub trait Sealed {}
}

/// Trait for types that can be converted from a fixed-size byte array with a specified endianness
#[unstable(feature = "read_le_be_internals", reason = "internals", issue = "none")]
// Once we can use associated consts in the types of method parameters, rewrite this to have
// `from_le_bytes` and `from_be_bytes` methods, move it to `core`, and make it public.
pub trait FromEndianBytes: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn read_le_from(r: &mut impl Read) -> Result<Self>;

    #[doc(hidden)]
    fn read_be_from(r: &mut impl Read) -> Result<Self>;
}

macro_rules! impl_from_endian_bytes {
    ($($t:ty),*$(,)?) => {$(
        #[unstable(feature = "sealed", issue = "none")]
        impl sealed::Sealed for $t {}

        #[unstable(feature = "read_le_be_internals", reason = "internals", issue = "none")]
        impl FromEndianBytes for $t {
            #[inline]
            fn read_le_from(r: &mut impl Read) -> Result<Self> {
                Ok(<$t>::from_le_bytes(r.read_array()?))
            }

            #[inline]
            fn read_be_from(r: &mut impl Read) -> Result<Self> {
                Ok(<$t>::from_be_bytes(r.read_array()?))
            }
        }
    )*};
}

impl_from_endian_bytes!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);
