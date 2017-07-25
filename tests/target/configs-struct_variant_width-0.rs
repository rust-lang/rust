// rustfmt-struct_variant_width: 0

// Force vertical layout when struct_variant_width is set to 0.

enum State {
    TryRecv {
        pos: usize,
        lap: u8,
        closed_count: usize,
    },
    Subscribe {
        pos: usize,
    },
    IsReady {
        pos: usize,
        ready: bool,
    },
    Unsubscribe {
        pos: usize,
        lap: u8,
        id_woken: usize,
    },
    FinalTryRecv {
        pos: usize,
        id_woken: usize,
    },
    TimedOut,
    Disconnected,
}
