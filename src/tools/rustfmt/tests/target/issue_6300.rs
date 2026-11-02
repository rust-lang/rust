const B: b = b {
    c: d,
    /* block comment, has bug */
    ..
};

const C: c = c {
    c: d,
    // line comment has bug
    ..
};
