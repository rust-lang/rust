// rustfmt-format_macro_bodies: true
// rustfmt-max_width: 82
// rustfmt-fn_call_width: 76

macro_rules! short {
    () => {
        m!(a, b, c);
    };
}

macro_rules! medium {
    () => {
        m!(aaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccccc, ddddddddd);
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
