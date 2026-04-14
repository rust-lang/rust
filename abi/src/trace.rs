#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TraceEvent {
    Empty,
    TimerTick { timestamp: u64 },
    Irq { vector: u8, timestamp: u64 },
    ContextSwitch { from: u64, to: u64, timestamp: u64 },
    PreemptDisable { depth: u32, timestamp: u64 },
}
