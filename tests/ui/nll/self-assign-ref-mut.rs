// Check that `*y` isn't borrowed after `y = y`.

//@ check-pass

fn main() {
    let mut x = 1;
    {
        let mut y = &mut x;
        y = y;
        y;
    }
    x;
    {
        let mut y = &mut x;
        y = y;
        y = y;
        y;
    }
    x;
}
