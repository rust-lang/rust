fn main() {
    let _ = [(); {
        //~^ WARNING Constant evaluating a complex constant, this might take some time
        let mut n = 0;
        while n < 5 {
            //~^ ERROR `while` is not allowed in a `const`
            //~| ERROR evaluation of constant value failed
            n = (n + 1) % 5;

            // Materialize a new AllocId. We need to make sure that `px` doesn't get promoted out.
            let x = [0isize; 4];
            let px = &x;
            n += px[0];
        }
        0
    }];
}
