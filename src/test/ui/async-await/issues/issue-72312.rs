// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

// edition:2018
fn require_static<T: 'static>(val: T) -> T {
    //[base]~^ NOTE 'static` lifetime requirement introduced by this bound
    val
}

struct Problem;

impl Problem {
    pub async fn start(&self) {
        //[base]~^ ERROR E0759
        //[base]~| NOTE this data with an anonymous lifetime `'_`
        //[base]~| NOTE in this expansion of desugaring of `async` block or function
        //[nll]~^^^^ NOTE let's call
        //[nll]~| NOTE `self` is a reference
        require_static(async move {
            //[base]~^ NOTE ...and is required to live as long as `'static` here
            //[nll]~^^ ERROR borrowed data escapes
            //[nll]~| NOTE `self` escapes
            //[nll]~| NOTE argument requires
            &self; //[base]~ NOTE ...is used here...
        });
    }
}

fn main() {}
