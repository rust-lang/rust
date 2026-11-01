// rustfmt-format_macro_bodies: true

macro_rules! short {
    () => {
        m!(a, b, c);
    };
}

macro_rules! medium {
    () => {
        m!(aaaaaaaaaa, bbbbbbbbbb, cccccccccc, ddddddddd);
    };
}

macro_rules! long {
    () => {
        m!(
            aaaaaaaaaaaaaaaa,
            bbbbbbbbbbbbbbbb,
            cccccccccccccccc,
            ddddddddddddddd
        );
    };
}
