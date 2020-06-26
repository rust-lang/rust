#![feature(const_eval_limit)]
#![const_eval_limit = "500"]

const X: usize = {
    let mut x = 0;
    while x != 1000 {
        //~^ ERROR any use of this value will cause an error
        x += 1;
    }

    x
};

fn main() {
    assert_eq!(X, 1000);
}
