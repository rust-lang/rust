//! This test checks that external macros don't hide
//! the const eval timeout lint and then subsequently
//! ICE.

//@compile-flags: --crate-type=lib -Ztiny-const-eval-limit
//@error-in-other-file: constant evaluation is taking a long time

static ROOK_ATTACKS_TABLE: () = {
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
    0_u64.count_ones();
};
