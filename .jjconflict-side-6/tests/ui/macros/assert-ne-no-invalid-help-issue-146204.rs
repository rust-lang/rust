macro_rules! local_assert_ne {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    //~^ ERROR can't compare `[u8; 4]` with `&[u8; 4]`
                    panic!();
                }
            }
        }
    };
}

fn main() {
    let buf = [0_u8; 4];
    assert_ne!(buf, b"----");
    //~^ ERROR can't compare `[u8; 4]` with `&[u8; 4]`

    assert_eq!(buf, b"----");
    //~^ ERROR can't compare `[u8; 4]` with `&[u8; 4]`

    local_assert_ne!(buf, b"----");
}
