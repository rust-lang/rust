// rustfmt-style_edition: 2024

// #2652
// Preserve trailing comma inside macro, even if it looks an array.
macro_rules! bar {
    ($m:ident) => {
        $m!([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z,]);
        $m!([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]);
    };
}
