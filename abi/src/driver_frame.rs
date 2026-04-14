use crate::display_driver_protocol as drvproto;
use crate::display_driver_protocol::DriverHeader;

pub type ParsedMessage<'a> = (DriverHeader, &'a [u8]);

pub struct FrameReader<const N: usize> {
    buf: [u8; N],
    head: usize,
    tail: usize,
    len: usize,
    dropped_bytes: usize,
    scratch: [u8; N],
}

impl<const N: usize> FrameReader<N> {
    pub const fn new() -> Self {
        Self {
            buf: [0u8; N],
            head: 0,
            tail: 0,
            len: 0,
            dropped_bytes: 0,
            scratch: [0u8; N],
        }
    }

    pub fn dropped_bytes(&self) -> usize {
        self.dropped_bytes
    }

    pub fn push(&mut self, data: &[u8]) -> usize {
        let free = N.saturating_sub(self.len);
        let to_write = data.len().min(free);
        let dropped = data.len() - to_write;
        if dropped > 0 {
            self.dropped_bytes = self.dropped_bytes.saturating_add(dropped);
        }

        if to_write == 0 {
            return 0;
        }

        let first = (N - self.tail).min(to_write);
        self.buf[self.tail..self.tail + first].copy_from_slice(&data[..first]);
        let remaining = to_write - first;
        if remaining > 0 {
            self.buf[0..remaining].copy_from_slice(&data[first..first + remaining]);
        }
        self.tail = (self.tail + to_write) % N;
        self.len += to_write;
        to_write
    }

    pub fn next_message(&mut self) -> Option<ParsedMessage<'_>> {
        loop {
            if self.len < drvproto::HEADER_SIZE {
                return None;
            }

            let mut header_buf = [0u8; drvproto::HEADER_SIZE];
            self.copy_out(0, &mut header_buf);
            let total = match drvproto::message_total_len(&header_buf) {
                Some(total) => total,
                None => {
                    self.advance_head(1);
                    continue;
                }
            };

            if total > N {
                self.advance_head(1);
                continue;
            }

            if self.len < total {
                return None;
            }

            if self.is_contiguous(total) {
                let slice =
                    unsafe { core::slice::from_raw_parts(self.buf.as_ptr().add(self.head), total) };
                let (header, payload_len) = match drvproto::parse_message(slice) {
                    Some((header, payload)) => (header, payload.len()),
                    None => {
                        self.advance_head(1);
                        continue;
                    }
                };
                let payload_ptr =
                    unsafe { self.buf.as_ptr().add(self.head + drvproto::HEADER_SIZE) };
                let payload = unsafe { core::slice::from_raw_parts(payload_ptr, payload_len) };
                self.advance_head(total);
                return Some((header, payload));
            } else {
                self.fill_scratch(total);
                let (header, payload_len) = match drvproto::parse_message(&self.scratch[..total]) {
                    Some((header, payload)) => (header, payload.len()),
                    None => {
                        self.advance_head(1);
                        continue;
                    }
                };
                let payload_ptr = unsafe { self.scratch.as_ptr().add(drvproto::HEADER_SIZE) };
                let payload = unsafe { core::slice::from_raw_parts(payload_ptr, payload_len) };
                self.advance_head(total);
                return Some((header, payload));
            }
        }
    }

    fn is_contiguous(&self, total: usize) -> bool {
        self.head + total <= N
    }

    fn copy_out(&self, offset: usize, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        let mut idx = (self.head + offset) % N;
        for byte in dst.iter_mut() {
            *byte = self.buf[idx];
            idx += 1;
            if idx == N {
                idx = 0;
            }
        }
    }

    fn fill_scratch(&mut self, total: usize) {
        if total == 0 {
            return;
        }
        let mut idx = self.head;
        for i in 0..total {
            self.scratch[i] = self.buf[idx];
            idx += 1;
            if idx == N {
                idx = 0;
            }
        }
    }

    fn advance_head(&mut self, count: usize) {
        if count >= self.len {
            self.head = self.tail;
            self.len = 0;
            return;
        }
        self.head = (self.head + count) % N;
        self.len -= count;
    }
}
