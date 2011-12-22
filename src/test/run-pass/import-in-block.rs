use std;

fn main() {
    import vec;
    import vec::to_mut;
    log_full(core::debug, vec::len(to_mut([1, 2])));
    {
        import vec::*;
        log_full(core::debug, len([2]));
    }
}
