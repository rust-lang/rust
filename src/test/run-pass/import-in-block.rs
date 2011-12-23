use std;

fn main() {
    import vec;
    import vec::to_mut;
    log(debug, vec::len(to_mut([1, 2])));
    {
        import vec::*;
        log(debug, len([2]));
    }
}
