// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

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
        //[base]~^ ERROR E0759
        foo(|| self.bar()).await;
        //[nll]~^ ERROR closure may outlive the current function
        //[nll]~| ERROR borrowed data escapes outside of associated function
    }

    pub fn bar(&self) {}
}

fn main() {}
