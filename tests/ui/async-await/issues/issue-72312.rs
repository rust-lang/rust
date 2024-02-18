//@ edition:2018
fn require_static<T: 'static>(val: T) -> T {
    val
}

struct Problem;

impl Problem {
    pub async fn start(&self) {
        //~^ NOTE let's call
        //~| NOTE `self` is a reference
        require_static(async move {
            //~^ ERROR borrowed data escapes
            //~| NOTE `self` escapes
            //~| NOTE argument requires
            &self;
        });
    }
}

fn main() {}
