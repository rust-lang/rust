fn main() {
    let mut x = 4;

    for uint::range(0, 3) |i| {
        // ensure that the borrow in this alt
	// does not inferfere with the swap
	// below.  note that it would it you
	// naively borrowed &x for the lifetime
	// of the variable x, as we once did
        match i {
          i => {
            let y = &x;
            assert i < *y;
          }
        }
        let mut y = 4;
        y <-> x;
    }
}
