//! An implementation of Pipes using UEFI variables

use super::common;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::ptr::NonNull;

pub struct AnonPipe {
    _pipe_data: Option<Box<uefi_pipe_protocol::Pipedata>>,
    _protocol: Option<common::ProtocolWrapper<uefi_pipe_protocol::Protocol>>,
    handle: NonNull<crate::ffi::c_void>,
}

unsafe impl Send for AnonPipe {}

// Safety: There are no threads in UEFI
unsafe impl Sync for AnonPipe {}

impl AnonPipe {
    pub(crate) fn new(
        pipe_data: Option<Box<uefi_pipe_protocol::Pipedata>>,
        protocol: Option<common::ProtocolWrapper<uefi_pipe_protocol::Protocol>>,
        handle: NonNull<crate::ffi::c_void>,
    ) -> Self {
        Self { _pipe_data: pipe_data, _protocol: protocol, handle }
    }

    pub(crate) fn null() -> Self {
        let pipe = common::ProtocolWrapper::install_protocol(uefi_pipe_protocol::Protocol::null())
            .unwrap();
        let handle = pipe.handle();
        Self::new(None, Some(pipe), handle)
    }

    pub(crate) fn make_pipe() -> Self {
        const MIN_BUFFER: usize = 1024;
        let mut pipe_data = Box::new(uefi_pipe_protocol::Pipedata::with_capacity(MIN_BUFFER));
        let pipe = common::ProtocolWrapper::install_protocol(
            uefi_pipe_protocol::Protocol::with_data(&mut pipe_data),
        )
        .unwrap();
        let handle = pipe.handle();
        Self::new(Some(pipe_data), Some(pipe), handle)
    }

    pub(crate) fn handle(&self) -> NonNull<crate::ffi::c_void> {
        self.handle
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let protocol = common::open_protocol(self.handle, uefi_pipe_protocol::PROTOCOL_GUID)?;
        unsafe { uefi_pipe_protocol::Protocol::read(protocol.as_ptr(), buf) }
    }

    pub(crate) fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let protocol = common::open_protocol(self.handle, uefi_pipe_protocol::PROTOCOL_GUID)?;
        unsafe { uefi_pipe_protocol::Protocol::read_to_end(protocol.as_ptr(), buf) }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let protocol = common::open_protocol(self.handle, uefi_pipe_protocol::PROTOCOL_GUID)?;
        unsafe { uefi_pipe_protocol::Protocol::write(protocol.as_ptr(), buf) }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn diverge(&self) -> ! {
        unimplemented!()
    }
}

pub fn read2(p1: AnonPipe, v1: &mut Vec<u8>, p2: AnonPipe, v2: &mut Vec<u8>) -> io::Result<()> {
    p1.read_to_end(v1)?;
    p2.read_to_end(v2)?;
    Ok(())
}

pub(crate) mod uefi_pipe_protocol {
    use crate::collections::VecDeque;
    use crate::io;
    use crate::sys::uefi::common;
    use io::{Read, Write};
    use r_efi::efi::Guid;

    pub(crate) const PROTOCOL_GUID: Guid = Guid::from_fields(
        0x3c4acb49,
        0xfb4c,
        0x45fb,
        0x93,
        0xe4,
        &[0x63, 0x5d, 0x71, 0x48, 0x4c, 0x0f],
    );

    // Maybe the internal data needs to be wrapped in a Mutex?
    #[repr(C)]
    #[derive(Debug)]
    pub(crate) struct Pipedata {
        data: VecDeque<u8>,
    }

    impl Pipedata {
        #[inline]
        pub(crate) fn with_capacity(capacity: usize) -> Pipedata {
            Pipedata { data: VecDeque::with_capacity(capacity) }
        }

        #[inline]
        pub(crate) unsafe fn read(data: *mut Pipedata, buf: &mut [u8]) -> io::Result<usize> {
            unsafe { (*data).data.read(buf) }
        }

        #[inline]
        pub(crate) unsafe fn read_to_end(
            data: *mut Pipedata,
            buf: &mut Vec<u8>,
        ) -> io::Result<usize> {
            unsafe { (*data).data.read_to_end(buf) }
        }

        #[inline]
        pub(crate) unsafe fn write(data: *mut Pipedata, buf: &[u8]) -> io::Result<usize> {
            unsafe { (*data).data.write(buf) }
        }
    }

    #[repr(C)]
    pub(crate) struct Protocol {
        data: *mut Pipedata,
    }

    impl common::Protocol for Protocol {
        const PROTOCOL_GUID: Guid = PROTOCOL_GUID;
    }

    impl Protocol {
        #[inline]
        pub(crate) fn with_data(data: &mut Pipedata) -> Self {
            Self { data }
        }

        #[inline]
        pub(crate) fn null() -> Self {
            Self { data: crate::ptr::null_mut() }
        }

        pub(crate) unsafe fn read(protocol: *mut Protocol, buf: &mut [u8]) -> io::Result<usize> {
            if unsafe { (*protocol).data.is_null() } {
                Ok(0)
            } else {
                unsafe { Pipedata::read((*protocol).data, buf) }
            }
        }

        pub(crate) unsafe fn read_to_end(
            protocol: *mut Protocol,
            buf: &mut Vec<u8>,
        ) -> io::Result<usize> {
            if unsafe { (*protocol).data.is_null() } {
                Ok(0)
            } else {
                unsafe { Pipedata::read_to_end((*protocol).data, buf) }
            }
        }

        pub(crate) unsafe fn write(protocol: *mut Protocol, buf: &[u8]) -> io::Result<usize> {
            if unsafe { (*protocol).data.is_null() } {
                Ok(buf.len())
            } else {
                unsafe { Pipedata::write((*protocol).data, buf) }
            }
        }
    }
}
