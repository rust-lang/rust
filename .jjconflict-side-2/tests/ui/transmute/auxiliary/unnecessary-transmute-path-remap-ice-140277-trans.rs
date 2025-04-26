//@ compile-flags: --remap-path-prefix=/=/non-existent
// helper for ../unnecessary-transmute-path-remap-ice-140277.rs

#[macro_export]
macro_rules! transmute {
    ($e:expr) => {{
        let e = $e;
        std::mem::transmute(e)
    }};
}
