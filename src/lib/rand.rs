


/**
 * Bindings the runtime's random number generator (ISAAC).
 */
native "rust" mod rustrt {
    type rctx;
    fn rand_new() -> rctx;
    fn rand_next(c: rctx) -> u32;
    fn rand_free(c: rctx);
}

type rng =
    obj {
        fn next() -> u32 ;
    };

fn mk_rng() -> rng {
    obj rt_rng(c: rustrt::rctx) {
        fn next() -> u32 {
            ret rustrt::rand_next(c);
        }
        drop { rustrt::rand_free(c); }
    }
    ret rt_rng(rustrt::rand_new());
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
