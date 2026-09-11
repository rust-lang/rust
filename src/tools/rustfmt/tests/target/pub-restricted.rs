pub(super) enum WriteState<D> {
    WriteId {
        id: U64Writer,
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteSize {
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteData(Writer<D>),
}

pub(crate) enum WriteState<D> {
    WriteId {
        id: U64Writer,
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteSize {
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteData(Writer<D>),
}

pub(in global::path::to::some_mod) enum WriteState<D> {
    WriteId {
        id: U64Writer,
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteSize {
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteData(Writer<D>),
}

pub(in local::path::to::some_mod) enum WriteState<D> {
    WriteId {
        id: U64Writer,
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteSize {
        size: U64Writer,
        payload: Option<Writer<D>>,
    },
    WriteData(Writer<D>),
}
