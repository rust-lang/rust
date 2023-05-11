// edition:2018
async fn foo<F>(fun: F)
where
    F: FnOnce() + 'static
{
    fun()
}

struct Struct;

impl Struct {
    pub async fn run_dummy_fn(&self) {
        foo(|| self.bar()).await;
        //~^ ERROR closure may outlive the current function
        //~| ERROR borrowed data escapes outside of method
    }

    pub fn bar(&self) {}
}

fn main() {}
