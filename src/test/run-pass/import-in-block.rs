use std;

fn main() {
    import vec;
    import vec::to_mut;
    log vec::len(to_mut([1, 2]));
    {
        import vec::*;
        log len([2]);
    }
}
