


// -*- rust -*-
pub fn main() {
    let mut i: int = 90;
    while i < 100 {
        info!("{}", i);
        i = i + 1;
        if i == 95 {
            let _v: ~[int] =
                ~[1, 2, 3, 4, 5]; // we check that it is freed by break

            info!("breaking");
            break;
        }
    }
    assert_eq!(i, 95);
}
