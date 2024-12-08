fn main() {
    let mut x = 0;
    || {
        || {
        //~^ ERROR captured variable cannot escape `FnMut` closure body
            let _y = &mut x;
        }
    };
}
