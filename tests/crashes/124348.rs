//@ known-bug: #124348
enum Eek {
    TheConst,
    UnusedByTheConst(Sum),
}

const EEK_ZERO: &[Eek] = &[];
