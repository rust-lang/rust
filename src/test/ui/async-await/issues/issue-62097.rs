// edition:2018
async fn foo<F>(fun: F)
where
    F: FnOnce() + 'static
{
    fun()
}

struct Struct;

impl Struct {
    pub async fn run_dummy_fn(&self) { //~ ERROR cannot infer
        foo(|| self.bar()).await;
    }

    pub fn bar(&self) {}
}

fn main() {}
