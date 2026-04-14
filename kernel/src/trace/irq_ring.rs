use abi::trace::TraceEvent;
use spin::Mutex;

const RING_SIZE: usize = 2048;

pub struct IrqRing {
    buffer: [TraceEvent; RING_SIZE],
    head: usize,
}

impl IrqRing {
    pub const fn new() -> Self {
        Self {
            buffer: [TraceEvent::Empty; RING_SIZE],
            head: 0,
        }
    }

    pub fn push_internal(&mut self, event: TraceEvent) {
        self.buffer[self.head] = event;
        self.head = (self.head + 1) % RING_SIZE;
    }

    pub fn read_all(&self, out: &mut [TraceEvent]) -> usize {
        let mut written = 0;
        let start = self.head;

        // Iterate from oldest (head) to newest (head-1)
        // Part 1: start to end
        for i in start..RING_SIZE {
            if i >= self.buffer.len() {
                break;
            }
            if let TraceEvent::Empty = self.buffer[i] {
                continue;
            }
            if written >= out.len() {
                break;
            }
            out[written] = self.buffer[i];
            written += 1;
        }

        // Part 2: 0 to start
        for i in 0..start {
            if let TraceEvent::Empty = self.buffer[i] {
                continue;
            }
            if written >= out.len() {
                break;
            }
            out[written] = self.buffer[i];
            written += 1;
        }

        written
    }
}

pub static IRQ_RING: Mutex<IrqRing> = Mutex::new(IrqRing::new());

pub fn push(event: TraceEvent) {
    if let Some(mut ring) = IRQ_RING.try_lock() {
        ring.push_internal(event);
    }
}
