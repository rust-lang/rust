// edition:2018
pub trait ParallelStream {
    async fn reduce(&self) {
//~^ ERROR functions in traits cannot be declared `async`
        vec![].into_iter().for_each(|_: ()| {})
    }
}

fn main() {}
