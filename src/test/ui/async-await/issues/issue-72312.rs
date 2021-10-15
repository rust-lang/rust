// edition:2018
fn require_static<T: 'static>(val: T) -> T {
    //~^ NOTE 'static` lifetime requirement introduced by this bound
    val
}

struct Problem;

impl Problem {
    pub async fn start(&self) { //~ ERROR E0759
        //~^ NOTE this data with an anonymous lifetime `'_`
        //~| NOTE in this expansion of desugaring of `async` block or function
        require_static(async move { //~ NOTE ...and is required to live as long as `'static` here
            &self; //~ NOTE ...is used here...
        });
    }
}

fn main() {}
