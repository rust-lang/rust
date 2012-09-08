use std;

fn main() {
    use vec::to_mut;
    log(debug, vec::len(to_mut(~[1, 2])));
    {
        use vec::*;
        log(debug, len(~[2]));
    }
}
