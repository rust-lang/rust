//@ edition: 2024

// Regression test for #140583. We want to borrowck nested
// bodies even if they are in dead code. While not necessary for
// soundness, it is desirable to error in such cases.

fn main() {
    return;
    |x: &str| -> &'static str { x };
    //~^ ERROR lifetime may not live long enough
    || {
        || {
            let temp = 1;
            let p: &'static u32 = &temp;
            //~^ ERROR `temp` does not live long enough
        };
    };
    const {
        let temp = 1;
        let p: &'static u32 = &temp;
        //~^ ERROR `temp` does not live long enough
    };
    async {
        let temp = 1;
        let p: &'static u32 = &temp;
        //~^ ERROR `temp` does not live long enough
    };
}
